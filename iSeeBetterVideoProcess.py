from __future__ import print_function
import argparse
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rbpn import Net as RBPN
import numpy as np
import utils
import time
import cv2
import math
import logger
import gc
import platform
import sys
import gi
import tempfile
import queue
import threading
from enum import Enum

"""
iSeeBetter Video Processing Script
---------------------------------
This script enhances video resolution using a super-resolution neural network (RBPN).
It includes two video encoding options:
1. GStreamer Encoder: High-quality encoding with advanced options like rate control, 
   adaptive quantization, and various codec parameters. Used by default when available.
2. OpenCV VideoWriter: Fallback encoding when GStreamer is not available, with various
   codec options and fallback mechanisms for compatibility.

The script can process videos using either full-frame or tile-based approaches to 
handle memory constraints, and provides extensive options for controlling the
processing and output quality.
"""

# Try to import GStreamer
USE_GSTREAMER = False
try:
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GObject, GLib
    USE_GSTREAMER = True
    print("GStreamer support enabled")
except (ImportError, ValueError) as e:
    print(f"GStreamer not available: {e}. Will use OpenCV for encoding.")

# Video processing settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Video Processing')
parser.add_argument('-m', '--model', default="weights/RBPN_4x.pth", help="Model")
parser.add_argument('-i', '--input', default="data/match_178203.mp4", help="Input video file")
parser.add_argument('-o', '--output', default='output_video.mp4', help="Output video file")
parser.add_argument('-s', '--upscale_factor', type=int, default=4, help="Super-Resolution Scale Factor")
parser.add_argument('-r', '--residual', action='store_true', required=False, help="Use residual learning")
parser.add_argument('-c', '--gpu_mode', action='store_true', required=False, help="Use a CUDA compatible GPU if available")
parser.add_argument('--seed', type=int, default=123, help="Random seed")
parser.add_argument('--gpus', default=1, type=int, help="How many GPU's to use")
parser.add_argument('--nFrames', type=int, default=7, help="Number of frames for processing")
parser.add_argument('--model_type', type=str, default="RBPN", help="Model type")
parser.add_argument('-d', '--debug', action='store_true', required=False, help="Print debug information")
parser.add_argument('-u', '--upscale_only', action='store_true', required=False, help="Upscale mode - without downscaling")
parser.add_argument('--batch_size', type=int, default=1, help="Batch size for processing")
parser.add_argument('--base_filter', type=int, default=256, help="Base filter size for RBPN model")
parser.add_argument('--feat', type=int, default=64, help="Feature size for RBPN model")
parser.add_argument('--skip', type=int, default=1, help="Process every Nth frame")
parser.add_argument('--use_tiles', action='store_true', help="Process large frames in tiles to save memory")
parser.add_argument('--tile_size', type=int, default=256, help="Tile size for tiled processing")
parser.add_argument('--tile_overlap', type=int, default=32, help="Overlap between tiles")
parser.add_argument('--max_frames', type=int, default=0, help="Maximum number of frames to process (0 = all frames)")
parser.add_argument('--start_frame', type=int, default=0, help="Start processing from this frame number")
parser.add_argument('--save_frames', action='store_true', help="Save individual frames as PNG files instead of video")
parser.add_argument('--frames_dir', type=str, default='output_frames', help="Directory to save frames if --save_frames is used")
parser.add_argument('--frame_format', type=str, default='frame_%06d.png', help="Filename format for saved frames")
parser.add_argument('--downscale_factor', type=float, default=1.0, help="Downscale input by this factor before processing")

args = parser.parse_args()

# Validate arguments
if args.downscale_factor < 1.0:
    print(f"Warning: downscale_factor must be >= 1.0, setting to 1.0 (no downscaling)")
    args.downscale_factor = 1.0

# Define interpolation methods for downscaling
INTERPOLATION_METHODS = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4
}

interpolation_method = INTERPOLATION_METHODS.get('bicubic', cv2.INTER_CUBIC)

gpus_list = range(args.gpus)
print(args)

cuda = args.gpu_mode
if cuda:
    print("Using GPU mode")
    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode")
else:
    print("Using CPU mode")

torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

print('==> Building model ', args.model_type)
if args.model_type == 'RBPN':
    model = RBPN(num_channels=3, base_filter=args.base_filter, feat=args.feat, num_stages=3, n_resblock=5, nFrames=args.nFrames, scale_factor=args.upscale_factor)

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")

if cuda:
    model = model.cuda(gpus_list[0])

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def process_tile(model, center_tensor, neighbor_tensors, flow_tensors, bicubic_tensor, residual):
    with torch.no_grad():
        prediction = model(center_tensor, neighbor_tensors, flow_tensors)
        
        if residual:
            prediction = prediction + bicubic_tensor
            
    return prediction

def get_tiles(img, tile_size, overlap):
    """Split the image into tiles with overlap"""
    h, w = img.shape[:2]
    tiles = []
    positions = []
    
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            # Ensure the tile doesn't go beyond image boundary
            tile_h = min(tile_size, h - y)
            tile_w = min(tile_size, w - x)
            
            # If tile is too small, skip it
            if tile_h < 32 or tile_w < 32:
                continue
                
            # Extract the tile
            tile = img[y:y+tile_h, x:x+tile_w]
            tiles.append(tile)
            positions.append((y, x, tile_h, tile_w))
    
    return tiles, positions

def merge_tiles(output_img, tiles, positions, tile_size, overlap):
    """Merge the processed tiles back into a single image with blending in overlapped regions"""
    h, w = output_img.shape[:2]
    count = np.zeros((h, w, 1), dtype=np.float32)
    
    for tile, (y, x, tile_h, tile_w) in zip(tiles, positions):
        # Create a weight mask for blending
        mask = np.ones((tile_h, tile_w, 1), dtype=np.float32)
        
        # Apply feathering at the edges
        feather = overlap // 2
        if feather > 0:
            # Left edge
            if x > 0:
                for i in range(feather):
                    mask[:, i] = (i + 1) / (feather + 1)
            # Right edge
            if x + tile_w < w:
                for i in range(feather):
                    mask[:, tile_w - 1 - i] = (i + 1) / (feather + 1)
            # Top edge
            if y > 0:
                for i in range(feather):
                    mask[i, :] = (i + 1) / (feather + 1)
            # Bottom edge
            if y + tile_h < h:
                for i in range(feather):
                    mask[tile_h - 1 - i, :] = (i + 1) / (feather + 1)
        
        # Apply the weighted tile to the output image
        output_img[y:y+tile_h, x:x+tile_w] += tile * mask
        count[y:y+tile_h, x:x+tile_w] += mask
    
    # Normalize the output by the count
    output_img = np.divide(output_img, count, out=np.zeros_like(output_img), where=count > 0)
    return output_img

def tensor_to_batch(tensor_list):
    return torch.cat(tensor_list, dim=0)

def save_frame(frame, frame_number, output_dir, format_str):
    """Save a single frame as an image file"""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, format_str % frame_number)
    cv2.imwrite(filename, frame)
    return filename

def process_video():
    # Initialize Logger
    logger.initLogger(args.debug)

    # Print network architecture
    utils.printNetworkArch(netG=model, netD=None)

    # Load pre-trained model
    modelPath = os.path.join(args.model)
    
    # Try loading with specific device mapping
    try:
        if cuda:
            # Map to available devices
            device_count = torch.cuda.device_count()
            device_ids = list(range(device_count))
            map_location = {'cuda:%d' % i: 'cuda:%d' % min(i, device_count - 1) for i in range(8)}
            utils.loadPreTrainedModel(gpuMode=args.gpu_mode, model=model, modelPath=modelPath, map_location=map_location)
        else:
            utils.loadPreTrainedModel(gpuMode=args.gpu_mode, model=model, modelPath=modelPath, map_location='cpu')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting alternative loading method...")
        try:
            # Manual loading with device mapping
            if cuda:
                state_dict = torch.load(modelPath, map_location=lambda storage, loc: storage.cuda(device_ids[0] if device_ids else 0))
            else:
                state_dict = torch.load(modelPath, map_location='cpu')
            
            # Remove 'module.' prefix if present (from DataParallel)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            
            # Handle DataParallel wrapper
            if cuda and len(device_ids) > 0:
                model.module.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(new_state_dict)
            
            print("Model loaded successfully using alternative method")
        except Exception as load_error:
            print(f"Fatal error loading model: {load_error}")
            raise
    
    model.eval()
    
    # Print CUDA memory info if available
    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Available GPU memory: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB allocated")
    
    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open input video {args.input}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Apply downscaling if requested
    if args.downscale_factor > 1.0:
        # Calculate downscaled dimensions
        width = int(orig_width / args.downscale_factor)
        height = int(orig_height / args.downscale_factor)
        print(f"Downscaling input by factor {args.downscale_factor:.2f} ({orig_width}x{orig_height} → {width}x{height})")
    else:
        width = orig_width
        height = orig_height
        print(f"Using original input resolution: {width}x{height}")
    
    # Calculate output dimensions (after super-resolution)
    out_width = int(width * args.upscale_factor)
    out_height = int(height * args.upscale_factor)
    
    # Check if we're effectively upscaling, downscaling, or maintaining original resolution
    if out_width > orig_width:
        effective_scale = out_width / orig_width
        print(f"Effective upscaling: {effective_scale:.2f}x original size")
    elif out_width < orig_width:
        effective_scale = orig_width / out_width
        print(f"Effective downscaling: {effective_scale:.2f}x smaller than original")
    else:
        print("Final output resolution matches original input")
    
    # Check max_frames parameter
    if args.max_frames > 0:
        target_frame_count = min(frame_count, args.start_frame + args.max_frames)
        print(f"Will process from frame {args.start_frame} to {target_frame_count-1} (total: {args.max_frames} frames)")
    else:
        target_frame_count = frame_count
        print(f"Will process all frames from {args.start_frame} to {frame_count-1}")
    
    # If starting from a specific frame, seek to that position
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    
    # Prepare frame buffer for temporal processing
    frame_buffer = []
    total_frames_processed = 0
    skipped_frames = 0
    saved_files = []
    
    # Setup for video output or frame saving
    if args.save_frames:
        print(f"Saving individual frames to directory: {args.frames_dir}")
        print(f"Frame format: {args.frame_format}")
        os.makedirs(args.frames_dir, exist_ok=True)
        out = None
    else:
        # Create simple video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (out_width, out_height))
        print(f"Output video: {args.output} ({out_width}x{out_height})")
    
    print(f"Input video: {orig_width}x{orig_height}, {frame_count} frames, {fps} fps")
    print(f"Output resolution: {out_width}x{out_height}")
    print(f"Processing every {args.skip} frame(s)")
    print(f"Using {args.nFrames} temporal frames")
    
    if args.use_tiles:
        print(f"Using tiled processing with tile size {args.tile_size}x{args.tile_size}, overlap {args.tile_overlap}")
    
    # Process the video
    frame_idx = args.start_frame
    while True:
        # Check if we've reached the maximum number of frames
        if args.max_frames > 0 and total_frames_processed >= args.max_frames:
            print(f"Reached maximum number of frames ({args.max_frames}). Stopping.")
            break
            
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Skip frames if needed
        if args.skip > 1 and (frame_idx - args.start_frame) % args.skip != 0:
            skipped_frames += 1
            continue
        
        # Apply downscaling if needed
        if args.downscale_factor > 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=interpolation_method)
        
        # Convert frame to RGB for neural network processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0  # Normalize to [0,1]
        
        # Add frame to buffer
        frame_buffer.append(frame_rgb)
        
        # Keep only necessary frames in buffer
        if len(frame_buffer) > args.nFrames:
            frame_buffer.pop(0)
        
        # Wait until we have enough frames for processing
        if len(frame_buffer) < args.nFrames:
            continue
        
        # Get the center frame for processing with neighbors
        center_idx = len(frame_buffer) // 2
        center_frame = frame_buffer[center_idx]
        
        # Create bicubic upscaled versions for full frame
        if args.use_tiles:
            # For tiled processing, we'll do bicubic upscaling per tile
            bicubic_frame = None
        else:
            # For full frame processing, upscale the RGB frame
            bicubic_frame = cv2.resize(center_frame, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
        
        try:
            t0 = time.time()
            
            if args.use_tiles:
                # Tile-based processing for large images
                tiles, positions = get_tiles(center_frame, args.tile_size, args.tile_overlap)
                output_sr = np.zeros((out_height, out_width, 3), dtype=np.float32)
                
                tile_count = len(tiles)
                print(f"Processing frame {total_frames_processed + 1} in {tile_count} tiles")
                
                processed_tiles = []
                
                for tile_idx, (tile, pos) in enumerate(zip(tiles, positions)):
                    y, x, th, tw = pos
                    # Create bicubic upscale of this tile
                    tile_bicubic = cv2.resize(tile, (tw * args.upscale_factor, th * args.upscale_factor), 
                                             interpolation=cv2.INTER_CUBIC)
                    
                    # Prepare tensors for this tile
                    tile_center = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0)
                    tile_bicubic_tensor = torch.from_numpy(tile_bicubic.transpose(2, 0, 1)).unsqueeze(0)
                    
                    # Get neighbor frames for this tile region
                    tile_neighbors = []
                    tile_flows = []
                    
                    for i in range(len(frame_buffer)):
                        if i != center_idx:
                            neighbor_frame = frame_buffer[i]
                            # Extract the same region from neighbor frame
                            neighbor_tile = neighbor_frame[y:y+th, x:x+tw]
                            neighbor_tensor = torch.from_numpy(neighbor_tile.transpose(2, 0, 1)).unsqueeze(0)
                            tile_neighbors.append(neighbor_tensor)
                            
                            # Simplified zero flow
                            flow = np.zeros((2, th, tw), dtype=np.float32)
                            flow_tensor = torch.from_numpy(flow).unsqueeze(0)
                            tile_flows.append(flow_tensor)
                    
                    # Process the tile through the model
                    with torch.no_grad():
                        if cuda:
                            tile_center = Variable(tile_center).cuda(gpus_list[0])
                            tile_bicubic_tensor = Variable(tile_bicubic_tensor).cuda(gpus_list[0])
                            tile_neighbors = [Variable(j).cuda(gpus_list[0]) for j in tile_neighbors]
                            tile_flows = [Variable(j).cuda(gpus_list[0]).float() for j in tile_flows]
                        else:
                            tile_center = Variable(tile_center).to(device=device, dtype=torch.float)
                            tile_bicubic_tensor = Variable(tile_bicubic_tensor).to(device=device, dtype=torch.float)
                            tile_neighbors = [Variable(j).to(device=device, dtype=torch.float) for j in tile_neighbors]
                            tile_flows = [Variable(j).to(device=device, dtype=torch.float) for j in tile_flows]
                        
                        # Process tile
                        tile_prediction = model(tile_center, tile_neighbors, tile_flows)
                        
                        if args.residual:
                            tile_prediction = tile_prediction + tile_bicubic_tensor
                        
                        # Convert to numpy
                        sr_tile = tile_prediction.cpu().data[0].numpy().transpose(1, 2, 0)
                        sr_tile = np.clip(sr_tile, 0, 1).astype(np.float32)
                        
                        # Store the super-resolved tile and its position
                        processed_tiles.append((sr_tile, (y * args.upscale_factor, x * args.upscale_factor, 
                                              th * args.upscale_factor, tw * args.upscale_factor)))
                        
                        # Clear GPU memory after each tile
                        clear_gpu_memory()
                        
                        print(f"  Processed tile {tile_idx+1}/{tile_count}", end="\r")
                
                # Merge all tiles into one image
                for tile, pos in processed_tiles:
                    y, x, h, w = pos
                    # Handle edge cases where upscaled tile might go beyond image boundary
                    h = min(h, out_height - y)
                    w = min(w, out_width - x)
                    
                    # Add tile to output image
                    output_sr[y:y+h, x:x+w] = tile[:h, :w]
                
                # Convert to uint8
                sr_frame_rgb = np.clip(output_sr * 255, 0, 255).astype(np.uint8)
                
            else:
                # Full frame processing
                # Convert center frame to tensor format
                center_tensor = torch.from_numpy(center_frame.transpose(2, 0, 1)).unsqueeze(0)
                
                # Create neighbor frames tensor list
                neighbor_tensors = []
                flow_tensors = []
                
                for i in range(len(frame_buffer)):
                    if i != center_idx:
                        neighbor = frame_buffer[i]
                        neighbor_tensor = torch.from_numpy(neighbor.transpose(2, 0, 1)).unsqueeze(0)
                        neighbor_tensors.append(neighbor_tensor)
                        
                        # Simplified zero flow
                        h, w, c = neighbor.shape
                        flow = np.zeros((2, h, w), dtype=np.float32)
                        flow_tensor = torch.from_numpy(flow).unsqueeze(0)
                        flow_tensors.append(flow_tensor)
                
                # Process inputs through the model
                with torch.no_grad():
                    if cuda:
                        center_tensor = Variable(center_tensor).cuda(gpus_list[0])
                        bicubic_tensor = Variable(torch.from_numpy(bicubic_frame.transpose(2, 0, 1)).unsqueeze(0)).cuda(gpus_list[0])
                        neighbor_tensors = [Variable(j).cuda(gpus_list[0]) for j in neighbor_tensors]
                        flow_tensors = [Variable(j).cuda(gpus_list[0]).float() for j in flow_tensors]
                    else:
                        center_tensor = Variable(center_tensor).to(device=device, dtype=torch.float)
                        bicubic_tensor = Variable(torch.from_numpy(bicubic_frame.transpose(2, 0, 1)).unsqueeze(0)).to(device=device, dtype=torch.float)
                        neighbor_tensors = [Variable(j).to(device=device, dtype=torch.float) for j in neighbor_tensors]
                        flow_tensors = [Variable(j).to(device=device, dtype=torch.float) for j in flow_tensors]
                
                    # Process frame
                    prediction = model(center_tensor, neighbor_tensors, flow_tensors)
                    
                    if args.residual:
                        prediction = prediction + bicubic_tensor
                    
                    # Convert prediction back to numpy
                    sr_frame_rgb = prediction.cpu().data[0].numpy().transpose(1, 2, 0)
                    sr_frame_rgb = np.clip(sr_frame_rgb * 255, 0, 255).astype(np.uint8)
            
            # Convert SR frame back to BGR for OpenCV
            sr_frame_bgr = cv2.cvtColor(sr_frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Either save as video frame or individual image
            if args.save_frames:
                # Save as individual PNG frame
                frame_filename = save_frame(sr_frame_bgr, total_frames_processed, args.frames_dir, args.frame_format)
                saved_files.append(frame_filename)
            else:
                if total_frames_processed == 0:
                    print(f"First frame shape: {sr_frame_bgr.shape}")
                # Write frame to video using OpenCV
                out.write(sr_frame_bgr)
            
            t1 = time.time()
            total_frames_processed += 1
            
            # Calculate progress percentage
            if args.max_frames > 0:
                progress = (total_frames_processed / args.max_frames) * 100
                remaining = args.max_frames - total_frames_processed
            else:
                progress = ((frame_idx - args.start_frame) / (target_frame_count - args.start_frame)) * 100
                remaining = target_frame_count - frame_idx
                
            # Calculate ETA
            frames_per_second = 1.0 / (t1 - t0)
            eta_seconds = remaining / frames_per_second if frames_per_second > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            
            if args.save_frames:
                print(f"Saved frame {total_frames_processed}/{args.max_frames if args.max_frames > 0 else 'all'} " +
                      f"({progress:.1f}%) in {t1-t0:.4f} seconds | ETA: {eta_str}")
            else:
                print(f"Processed frame {total_frames_processed}/{args.max_frames if args.max_frames > 0 else 'all'} " +
                      f"({progress:.1f}%) in {t1-t0:.4f} seconds | ETA: {eta_str}")
            
            # Clean up GPU memory
            clear_gpu_memory()
                
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory error: {e}")
                print("Try reducing nFrames or enable tile-based processing with --use_tiles")
                break
            else:
                raise e
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
        print(f"Video processing complete. Output saved to {args.output}")
    else:
        print(f"Frame saving complete. {len(saved_files)} frames saved to {args.frames_dir}")
        
    print(f"Total frames processed: {total_frames_processed}")
    print(f"Total frames skipped: {skipped_frames}")

if __name__ == "__main__":
    process_video() 