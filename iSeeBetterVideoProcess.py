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
# parser.add_argument('-m', '--model', default="weights/RBPN_2x.pth", help="Model")
# parser.add_argument('-m', '--model', default="weights/RBPN_4x_F11_NTIRE2019.pth", help="Model") # this is not working
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
parser.add_argument('--base_filter', type=int, default=256, help="Base filter size for RBPN model") # 640
parser.add_argument('--feat', type=int, default=64, help="Feature size for RBPN model")
parser.add_argument('--skip', type=int, default=1, help="Process every Nth frame")
parser.add_argument('--use_tiles', action='store_true', help="Process large frames in tiles to save memory")
parser.add_argument('--tile_size', type=int, default=256, help="Tile size for tiled processing")
parser.add_argument('--tile_overlap', type=int, default=32, help="Overlap between tiles")
parser.add_argument('--max_frames', type=int, default=0, help="Maximum number of frames to process (0 = all frames)")
parser.add_argument('--start_frame', type=int, default=0, help="Start processing from this frame number")
parser.add_argument('--codec', type=str, default='H264', help="Video codec (H264, HEVC, AV1, ProRes, or RAW)")
parser.add_argument('--bitrate', type=str, default='0', help="Video bitrate in Mbps (0 for auto/high quality)")
parser.add_argument('--crf', type=int, default=17, help="Constant Rate Factor (lower is better quality, 0-51)")
parser.add_argument('--use_gpu_encoder', action='store_true', help="Use NVIDIA GPU hardware encoder")
parser.add_argument('--save_frames', action='store_true', help="Save individual frames as PNG files instead of video")
parser.add_argument('--frames_dir', type=str, default='output_frames', help="Directory to save frames if --save_frames is used")
parser.add_argument('--frame_format', type=str, default='frame_%06d.png', help="Filename format for saved frames")
parser.add_argument('--downscale_factor', type=float, default=1.0, help="Downscale input by this factor before processing")
parser.add_argument('--downscale_method', type=str, default='bicubic', help="Downscaling method: nearest, bilinear, bicubic, lanczos")
parser.add_argument('--use_gstreamer', action='store_true', help="Use GStreamer for video encoding (enabled by default when available)")
parser.add_argument('--gst_preset', type=str, default='high-quality', help="GStreamer encoder preset (high-quality, low-latency, ultra-quality)")
parser.add_argument('--temporal_aq', action='store_true', help="Enable temporal adaptive quantization for better quality")
parser.add_argument('--spatial_aq', type=int, default=8, help="Spatial adaptive quantization (0-15, 0=off)")
parser.add_argument('--rc_mode', type=str, default='cbr', help="Rate control mode: cbr, vbr, or cqp")
parser.add_argument('--save_yuv_test', action='store_true', help="Save a test YUV frame to diagnose color issues")

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

if args.downscale_method.lower() not in INTERPOLATION_METHODS:
    print(f"Warning: Unknown downscale method '{args.downscale_method}', using 'bicubic'")
    args.downscale_method = 'bicubic'

interpolation_method = INTERPOLATION_METHODS[args.downscale_method.lower()]

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

def create_video_writer(output_file, fps, width, height):
    """Create a video writer with the best quality settings"""
    # Check if NVIDIA GPU encoder is available and requested
    use_gpu_encoder = args.use_gpu_encoder and cuda and torch.cuda.is_available()
    
    # Determine OS platform
    is_windows = platform.system() == 'Windows'
    is_linux = platform.system() == 'Linux'
    is_mac = platform.system() == 'Darwin'
    
    # Get file extension
    _, ext = os.path.splitext(output_file.lower())
    if not ext:
        ext = '.mp4'  # Default extension
        output_file += ext
    
    # Adjust output format based on codec if needed
    adjusted_path = False
    
    # Define fourcc codec based on selection and platform
    if args.codec.upper() == 'H264':
        if use_gpu_encoder and (is_windows or is_linux):
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            print("Using NVIDIA hardware-accelerated H264 codec")
        else:
            # Software H.264 encoder (widely supported)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            print("Using software H264/AVC codec")
    elif args.codec.upper() == 'HEVC' or args.codec.upper() == 'H265':
        if use_gpu_encoder and (is_windows or is_linux):
            fourcc = cv2.VideoWriter_fourcc(*'HEVC')
            print("Using NVIDIA hardware-accelerated HEVC codec")
        else:
            fourcc = cv2.VideoWriter_fourcc(*'hvc1')
            print("Using software HEVC/H265 codec")
    elif args.codec.upper() == 'AV1':
        # AV1 codec (may not be supported in all OpenCV builds)
        fourcc = cv2.VideoWriter_fourcc(*'av01')
        print("Using AV1 codec (may not be supported in all OpenCV builds)")
    elif args.codec.upper() == 'PRORES':
        # ProRes codec (best supported on Mac)
        fourcc = cv2.VideoWriter_fourcc(*'apch')
        print("Using ProRes codec (best on macOS)")
    elif args.codec.upper() == 'LOSSLESS':
        # Use PNG lossless encoding for MP4 container
        if ext == '.mp4':
            # MJPEG in MP4 for high quality (not truly lossless but high quality)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            print("Using MJPEG high quality codec in MP4 container")
        else:
            # FFV1 requires .avi container
            new_output = os.path.splitext(output_file)[0] + '.avi'
            output_file = new_output
            fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            print(f"Using FFV1 lossless codec (changed output to AVI: {output_file})")
            adjusted_path = True
    else:
        # Default to MP4V (widely supported)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("Using default MP4V codec")

    # Calculate target bitrate (bits per pixel * pixels per frame * frames per second)
    if args.bitrate != '0':
        try:
            bitrate_str = f"{float(args.bitrate):.1f} Mbps"
        except ValueError:
            bitrate_str = "auto (high quality)"
    else:
        bitrate_str = "auto (high quality)"
        
    print(f"Encoding with {args.codec.upper()} codec, bitrate: {bitrate_str}")
    print(f"Output file: {output_file}")
    
    # Create the video writer with the selected codec
    # Note: OpenCV doesn't expose all encoding parameters via the Python API
    # For better encoding quality, we'll try MP4V first if H264/HEVC fails
    try:
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Verify the writer is initialized
        if not out.isOpened():
            print(f"Warning: Couldn't create video with codec {args.codec.upper()}.")
            
            # Try another lossless option if lossless was requested
            if args.codec.upper() == 'LOSSLESS' and not adjusted_path:
                print("Trying alternative lossless format (AVI container with FFV1 codec)")
                new_output = os.path.splitext(output_file)[0] + '.avi'
                output_file = new_output
                fourcc = cv2.VideoWriter_fourcc(*'FFV1')
                out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
                
                if out.isOpened():
                    print(f"Successfully created lossless video with FFV1 codec in AVI container: {output_file}")
                    return out
            
            print("Falling back to default MP4V codec")
            # Fallback to default codec
            out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            if not out.isOpened():
                print("Failed to create video writer with MP4V codec.")
                print("Trying with MJPG codec (high quality but larger file size)")
                
                # Try MJPG as a last resort
                out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
                
                if not out.isOpened():
                    raise RuntimeError("Failed to create video writer with any codec. Check OpenCV installation.")
                else:
                    print("Successfully created video writer with MJPG codec")
    except Exception as e:
        print(f"Error creating VideoWriter: {e}")
        print("Trying with MJPG codec as fallback")
        
        # Try MJPG as fallback
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
        
        if not out.isOpened():
            print("Error: Could not create video writer with any codec.")
            # Create a low-quality but widely compatible FMP4 writer as last resort
            out = cv2.VideoWriter(output_file, 0, fps, (width, height))
    
    return out

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
    
    # Fix for loading models saved on multi-GPU systems
    try:
        # Try loading with specific device mapping
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
        print(f"Using {args.downscale_method} interpolation for downscaling")
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
    orig_frame_buffer = []  # Store original frames for color reference
    total_frames_processed = 0
    skipped_frames = 0
    saved_files = []
    encoder = None  # Initialize encoder to None
    
    # Setup for video output or frame saving
    if args.save_frames:
        print(f"Saving individual frames to directory: {args.frames_dir}")
        print(f"Frame format: {args.frame_format}")
        os.makedirs(args.frames_dir, exist_ok=True)
        out = None
    else:
        if USE_GSTREAMER:  # Use GStreamer whenever available
            # Use GStreamer encoder
            print("Using GStreamer for high quality encoding")
            if args.use_gpu_encoder:
                print("Attempting to use NVIDIA hardware encoding")
            
            encoder = create_gstreamer_encoder(args.output, fps, out_width, out_height)
            if encoder is None:
                # Fallback to OpenCV if GStreamer fails
                print("Falling back to OpenCV VideoWriter")
                out = create_video_writer(args.output, fps, out_width, out_height)
            else:
                out = None  # We'll use encoder.feed_frame() instead
        else:
            # Use OpenCV VideoWriter
            print("GStreamer not available, using OpenCV VideoWriter")
            out = create_video_writer(args.output, fps, out_width, out_height)
        
        print(f"Output file: {args.output}")
    
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
        
        # Keep a copy of the original frame for color reference
        original_frame = frame.copy()
        
        # Apply downscaling if needed
        if args.downscale_factor > 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=interpolation_method)
            original_frame = cv2.resize(original_frame, (width, height), interpolation=interpolation_method)
        
        # Store original frame for later color reference
        orig_frame_buffer.append(original_frame)
        
        # Convert frame to RGB for neural network processing
        # Neural networks typically work better with RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0  # Normalize to [0,1]
        
        # Add frame to buffer
        frame_buffer.append(frame_rgb)
        
        # Keep only necessary frames in buffer
        if len(frame_buffer) > args.nFrames:
            frame_buffer.pop(0)
            orig_frame_buffer.pop(0)
        
        # Wait until we have enough frames for processing
        if len(frame_buffer) < args.nFrames:
            continue
        
        # Get the center frame for processing with neighbors
        center_idx = len(frame_buffer) // 2
        center_frame = frame_buffer[center_idx]
        # center_orig = orig_frame_buffer[center_idx]
        
        # Create bicubic upscaled versions for full frame
        # Original frame (BGR) upscaled for color reference
        # orig_upscaled = cv2.resize(center_orig, (out_width, out_height), interpolation=cv2.INTER_CUBIC)
        
        # RGB frame upscaled for model processing if needed
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
            
            # Color correction to match original frame's color characteristics
            # Convert SR frame back to BGR for OpenCV
            sr_frame_bgr = cv2.cvtColor(sr_frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Either save as video frame or individual image
            if args.save_frames:
                # Save as individual PNG frame
                frame_filename = save_frame(sr_frame_bgr, total_frames_processed, args.frames_dir, args.frame_format)
            else:
                if total_frames_processed == 0:
                    print(f"Frame {frame_idx} is {sr_frame_bgr.shape} debug dumping.")
                     # Save as individual PNG frame
                    frame_filename = save_frame(sr_frame_bgr, total_frames_processed, args.frames_dir, args.frame_format)
                    
                # Write frame to video
                if encoder is not None:
                    # Use GStreamer encoder
                    if not encoder.feed_frame(sr_frame_bgr):
                        print("Error feeding frame to GStreamer encoder, switching to OpenCV VideoWriter")
                        # Close the GStreamer encoder
                        encoder.stop()
                        # Create OpenCV VideoWriter as fallback
                        out = create_video_writer(args.output, fps, out_width, out_height)
                        # Write the current frame
                        out.write(sr_frame_bgr)
                        # Set encoder to None to indicate we're now using OpenCV
                        encoder = None
                else:
                    # Use OpenCV writer
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
                # Clean up resources before exiting
                if encoder is not None:
                    encoder.stop()
                break
            else:
                raise e
    
    # Release resources
    cap.release()
    if encoder is not None:
        encoder.stop()
        print(f"Video processing complete. Output saved to {args.output}")
    elif out is not None:
        out.release()
        print(f"Video processing complete. Output saved to {args.output}")
    else:
        print(f"Frame saving complete. {len(saved_files)} frames saved to {args.frames_dir}")
        
    print(f"Total frames processed: {total_frames_processed}")
    print(f"Total frames skipped: {skipped_frames}")

# GStreamer implementation
class GstEncoder:
    """High quality video encoder using GStreamer with advanced encoding options"""
    
    class RateControlMode(Enum):
        CBR = 1  # Constant Bitrate
        VBR = 2  # Variable Bitrate
        CQP = 3  # Constant Quantization Parameter
    
    def __init__(self, output_file, width, height, fps, 
                 codec='H264', 
                 bitrate_mbps=0,
                 use_nvidia=False,
                 preset='high-quality',
                 rc_mode='cbr',
                 temporal_aq=True,
                 spatial_aq=8):
        """
        Initialize GStreamer encoder
        
        Args:
            output_file: Output file path
            width: Frame width
            height: Frame height
            fps: Frames per second
            codec: Video codec (H264, HEVC)
            bitrate_mbps: Bitrate in Mbps (0 for auto)
            use_nvidia: Use NVIDIA hardware encoder
            preset: Encoder preset (high-quality, low-latency, ultra-quality)
            rc_mode: Rate control mode (cbr, vbr, cqp)
            temporal_aq: Enable temporal adaptive quantization
            spatial_aq: Spatial adaptive quantization strength (0-15, 0=off)
        """
        if not USE_GSTREAMER:
            raise RuntimeError("GStreamer is not available. Cannot use GstEncoder.")
        
        # Initialize GStreamer if not already done
        if not Gst.is_initialized():
            Gst.init(None)
        
        self.output_file = output_file
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec.upper()
        self.use_nvidia = use_nvidia
        self.preset = preset
        
        # Set rate control mode
        if rc_mode.lower() == 'cbr':
            self.rc_mode = self.RateControlMode.CBR
        elif rc_mode.lower() == 'vbr':
            self.rc_mode = self.RateControlMode.VBR
        elif rc_mode.lower() == 'cqp':
            self.rc_mode = self.RateControlMode.CQP
        else:
            print(f"Unknown rate control mode: {rc_mode}. Using CBR.")
            self.rc_mode = self.RateControlMode.CBR
        
        # Calculate bitrate if not specified
        if bitrate_mbps <= 0:
            # Auto bitrate calculation: 0.1 bits per pixel for each frame
            self.bitrate = int(width * height * fps * 0.2)
        else:
            # Convert Mbps to bps
            self.bitrate = int(bitrate_mbps * 1024 * 1024)
        
        # Quality settings
        self.temporal_aq = temporal_aq
        self.spatial_aq = min(15, max(0, spatial_aq))  # Clamp to 0-15
        
        # Internal state
        self.pipeline = None
        self.appsrc = None
        self.loop = None
        self.frame_count = 0
        self.buffer_size = fps  # Buffer one second of frames
        self.frame_queue = queue.Queue(maxsize=self.buffer_size)
        self.is_running = False
        self.main_loop_thread = None
        self.error_message = None
        self.use_simple_pipeline = True  # Use the simplified pipeline approach
        
        # Try to create pipeline using alternative method first
        if self.use_simple_pipeline:
            if not self._create_simple_pipeline():
                print("Simple pipeline creation failed, falling back to element-by-element pipeline")
                self.use_simple_pipeline = False
                self._create_pipeline()
        else:
            # Create pipeline element by element
            self._create_pipeline()

    def _create_simple_pipeline(self):
        """Create a GStreamer pipeline using parse_launch for simpler configuration"""
        try:
            # Select encoder based on codec and hardware
            if self.codec == 'H264':
                if self.use_nvidia:
                    # Try NVIDIA encoder with specific settings
                    encoder_str = "nvv4l2h264enc"
                    if self.rc_mode == self.RateControlMode.CBR:
                        encoder_str += " rc-mode=1"  # CBR
                    elif self.rc_mode == self.RateControlMode.VBR:
                        encoder_str += " rc-mode=2"  # VBR
                    else:
                        encoder_str += " rc-mode=0"  # CQP
                    
                    if self.preset == 'high-quality':
                        encoder_str += " preset-id=1"
                    elif self.preset == 'low-latency':
                        encoder_str += " preset-id=2"
                    elif self.preset == 'ultra-quality':
                        encoder_str += " preset-id=3"
                        
                    if self.temporal_aq:
                        encoder_str += " temporal-aq=1"
                    
                    if self.spatial_aq > 0:
                        encoder_str += f" spatial-aq=1 aq-strength={self.spatial_aq}"
                        
                    encoder_str += f" bitrate={self.bitrate // 1000}"
                else:
                    # Software encoder
                    preset = "slower" if self.preset == 'high-quality' else "veryfast"
                    encoder_str = f"x264enc speed-preset={preset} bitrate={self.bitrate // 1000}"
            else:
                # HEVC
                if self.use_nvidia:
                    encoder_str = f"nvv4l2h265enc bitrate={self.bitrate // 1000}"
                else:
                    encoder_str = f"x265enc bitrate={self.bitrate // 1000}"
            
            # Determine muxer
            file_ext = os.path.splitext(self.output_file)[1].lower()
            if file_ext == '.mp4':
                muxer_str = "mp4mux"
            elif file_ext == '.mkv':
                muxer_str = "matroskamux"
            else:
                muxer_str = "mp4mux"  # Default to MP4
            
            # Try multiple pipeline configurations, from most optimal to most compatible
            pipeline_configs = []
            
            # NVIDIA hardware pipeline configurations
            if self.use_nvidia:
                # 1. First try with NV12 format in NVMM memory (best for NVIDIA hardware)
                pipeline_configs.append(
                    f"appsrc name=source format=time is-live=true do-timestamp=true "
                    f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                    f"videoconvert ! video/x-raw,format=NV12 ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
                    f"{encoder_str} ! {muxer_str} ! filesink location={self.output_file}"
                )
                
                # # 2. Try with BGR input and nvvidconv with memory features
                # pipeline_configs.append(
                #     f"appsrc name=source format=time is-live=true do-timestamp=true "
                #     f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                #     f"videoconvert ! nvvidconv ! "
                #     f"{encoder_str} ! {muxer_str} ! filesink location={self.output_file}"
                # )
                
                # # 3. Try with I420 format (common format for encoders)
                # pipeline_configs.append(
                #     f"appsrc name=source format=time is-live=true do-timestamp=true "
                #     f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                #     f"videoconvert ! video/x-raw,format=I420 ! nvvidconv ! "
                #     f"{encoder_str} ! {muxer_str} ! filesink location={self.output_file}"
                # )
                
                # # 4. Try with RGB format and explicit caps
                # pipeline_configs.append(
                #     f"appsrc name=source format=time is-live=true do-timestamp=true "
                #     f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                #     f"videoconvert ! video/x-raw,format=RGB ! nvvidconv ! video/x-raw(memory:NVMM) ! "
                #     f"{encoder_str} ! {muxer_str} ! filesink location={self.output_file}"
                # )
            else:
                # Software encoding pipeline
                pipeline_configs.append(
                    f"appsrc name=source format=time is-live=true do-timestamp=true "
                    f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1 ! "
                    f"videoconvert ! {encoder_str} ! {muxer_str} ! filesink location={self.output_file}"
                )
            
            # Try each configuration until one works
            for i, pipeline_str in enumerate(pipeline_configs):
                try:
                    print(f"Trying pipeline configuration {i+1}/{len(pipeline_configs)}:")
                    print(pipeline_str)
                    
                    # Create pipeline
                    self.pipeline = Gst.parse_launch(pipeline_str)
                    if not self.pipeline:
                        print(f"Failed to create pipeline with configuration {i+1}")
                        continue
                    
                    # Get appsrc element
                    self.appsrc = self.pipeline.get_by_name("source")
                    if not self.appsrc:
                        print(f"Failed to get appsrc element from pipeline configuration {i+1}")
                        continue
                    
                    # Set up bus to catch errors
                    bus = self.pipeline.get_bus()
                    bus.add_signal_watch()
                    bus.connect("message", self._on_message)
                    
                    print(f"Successfully created pipeline with configuration {i+1}")
                    return True
                except Exception as e:
                    print(f"Error creating pipeline with configuration {i+1}: {e}")
            
            # If we get here, none of the configurations worked
            print("All pipeline configurations failed")
            return False
            
        except Exception as e:
            print(f"Error creating simple pipeline: {e}")
            return False

    def _create_pipeline(self):
        """Create the GStreamer pipeline"""
        # Select encoder based on codec and hardware
        is_nvidia = self.use_nvidia
        if self.codec == 'H264':
            if is_nvidia:
                encoder_name = "nvv4l2h264enc"  # NVIDIA hardware H.264 encoder
            else:
                encoder_name = "x264enc"  # Software H.264 encoder
        elif self.codec == 'HEVC' or self.codec == 'H265':
            if is_nvidia:
                encoder_name = "nvv4l2h265enc"  # NVIDIA hardware HEVC encoder
            else:
                encoder_name = "x265enc"  # Software HEVC encoder
        else:
            raise ValueError(f"Unsupported codec for GStreamer: {self.codec}")
        
        # Create pipeline
        self.pipeline = Gst.Pipeline.new("encoder-pipeline")
        if not self.pipeline:
            raise RuntimeError("Failed to create GStreamer pipeline")
        
        # Create elements
        self.appsrc = Gst.ElementFactory.make("appsrc", "source")
        videorate = Gst.ElementFactory.make("videorate", "videorate")
        videoconvert1 = Gst.ElementFactory.make("videoconvert", "videoconvert1")
        
        # Different elements based on encoder type
        if is_nvidia:
            # Add NVIDIA-specific elements for hardware encoding
            try:
                # Try to create nvvidconv
                nvvidconv = Gst.ElementFactory.make("nvvidconv", "nvvidconv")
                if not nvvidconv:
                    print("nvvidconv element not available, falling back to software encoding")
                    is_nvidia = False
                    encoder_name = "x264enc" if self.codec == 'H264' else "x265enc"
                else:
                    print("Using NVIDIA hardware acceleration with nvvidconv")
                    
                    # Add a capsfilter for NV12 format which works better with NVIDIA encoders
                    caps_filter1 = Gst.ElementFactory.make("capsfilter", "caps_filter1")
                    caps1 = Gst.Caps.from_string("video/x-raw,format=NV12")
                    caps_filter1.set_property("caps", caps1)
                    
                    # Add a capsfilter for NVMM memory type for optimized memory handling
                    caps_filter2 = Gst.ElementFactory.make("capsfilter", "caps_filter2")
                    caps2 = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12")
                    caps_filter2.set_property("caps", caps2)
                    
                    # Check that capsfilters were created successfully
                    if not caps_filter1 or not caps_filter2:
                        print("Failed to create capsfilter, continuing without explicit format")
                    
            except Exception as e:
                print(f"Error creating NVIDIA elements: {e}, falling back to software encoding")
                is_nvidia = False
                encoder_name = "x264enc" if self.codec == 'H264' else "x265enc"
        
        # Create the encoder element
        encoder = Gst.ElementFactory.make(encoder_name, "encoder")
        if not encoder:
            print(f"Failed to create encoder {encoder_name}, falling back to x264enc")
            encoder_name = "x264enc"
            encoder = Gst.ElementFactory.make(encoder_name, "encoder")
            is_nvidia = False
            
        # Determine container format and muxer
        file_ext = os.path.splitext(self.output_file)[1].lower()
        if file_ext == '.mp4':
            muxer = Gst.ElementFactory.make("mp4mux", "muxer")
        elif file_ext == '.mkv':
            muxer = Gst.ElementFactory.make("matroskamux", "muxer")
        else:
            # Default to MP4
            muxer = Gst.ElementFactory.make("mp4mux", "muxer")
        
        filesink = Gst.ElementFactory.make("filesink", "filesink")
        
        # Check that essential elements were created
        for element in [self.appsrc, videorate, videoconvert1, encoder, muxer, filesink]:
            if not element:
                raise RuntimeError(f"Failed to create essential GStreamer element: {element.get_name()}")
        
        # Configure source
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("do-timestamp", True)
        
        # BGR format for input
        caps = Gst.Caps.from_string(f"video/x-raw,format=BGR,width={self.width},height={self.height},framerate={self.fps}/1")
        self.appsrc.set_property("caps", caps)
        
        # Configure videorate
        videorate.set_property("max-rate", self.fps)
        
        # Configure encoder
        encoder.set_property("bitrate", self.bitrate // 1000)  # Convert to Kbps
        
        if encoder_name.startswith("nvv4l2"):
            # NVIDIA hardware encoder properties
            # Set preset
            if self.preset == 'high-quality':
                encoder.set_property("preset-id", 1)  # High quality
            elif self.preset == 'low-latency':
                encoder.set_property("preset-id", 2)  # Low latency
            elif self.preset == 'ultra-quality':
                encoder.set_property("preset-id", 3)  # Ultra quality
            
            # Set rate control mode
            if self.rc_mode == self.RateControlMode.CBR:
                encoder.set_property("rc-mode", 1)  # CBR
            elif self.rc_mode == self.RateControlMode.VBR:
                encoder.set_property("rc-mode", 2)  # VBR
            elif self.rc_mode == self.RateControlMode.CQP:
                encoder.set_property("rc-mode", 0)  # CQP
                encoder.set_property("qp-range", "20,20:25,25:27,27")  # I,P,B frame QP values
            
            # Set quality options
            if self.temporal_aq:
                encoder.set_property("temporal-aq", True)
            
            if self.spatial_aq > 0:
                encoder.set_property("spatial-aq", True)
                encoder.set_property("aq-strength", self.spatial_aq)
            
            # Additional quality settings
            encoder.set_property("iframeinterval", 25)  # I-frame every 1 seconds (at 25 fps)
            encoder.set_property("insert-sps-pps", True)
            encoder.set_property("insert-vui", True)
            
        elif encoder_name == "x264enc":
            # x264 software encoder properties
            if self.preset == 'high-quality':
                encoder.set_property("speed-preset", "slower")
            elif self.preset == 'low-latency':
                encoder.set_property("speed-preset", "veryfast")
            elif self.preset == 'ultra-quality':
                encoder.set_property("speed-preset", "veryslow")
            
            # Use "film" tune only when supported by the GStreamer installation
            try:
                # Try to set film tune - this will fail if not supported
                valid_tunes = Gst.Element.get_property(encoder, "tune")
                if "film" in str(valid_tunes):
                    encoder.set_property("tune", "film")
                else:
                    print("GStreamer x264enc 'film' tune not supported. Using default tuning.")
            except:
                # Tune property might not exist or not be settable
                print("GStreamer x264enc 'tune' property not available. Using default tuning.")
                
            encoder.set_property("key-int-max", 25)  # I-frame every 1 seconds (at 25 fps)
            
            if self.rc_mode == self.RateControlMode.CQP:
                encoder.set_property("qp-min", 18)
                encoder.set_property("qp-max", 30)
                encoder.set_property("qp-step", 3)
            
        elif encoder_name == "x265enc":
            # x265 software encoder properties
            if self.preset == 'high-quality':
                encoder.set_property("speed-preset", "slower")
            elif self.preset == 'low-latency':
                encoder.set_property("speed-preset", "veryfast")
            elif self.preset == 'ultra-quality':
                encoder.set_property("speed-preset", "veryslow")
            
            # Use "grain" tune only when supported by the GStreamer installation
            try:
                valid_tunes = Gst.Element.get_property(encoder, "tune")
                if "grain" in str(valid_tunes):
                    encoder.set_property("tune", "grain")
                else:
                    print("GStreamer x265enc 'grain' tune not supported. Using default tuning.")
            except:
                print("GStreamer x265enc 'tune' property not available. Using default tuning.")
                
            encoder.set_property("key-int-max", 60)  # I-frame every 2 seconds (at 30 fps)
        
        # Configure sink
        filesink.set_property("location", self.output_file)
        
        # Add elements to pipeline
        self.pipeline.add(self.appsrc)
        self.pipeline.add(videorate)
        self.pipeline.add(videoconvert1)
        
        # Add NVIDIA conversion if hardware encoding
        if is_nvidia:
            # Add capsfilter for NV12 format if available
            if 'caps_filter1' in locals():
                self.pipeline.add(caps_filter1)
            
            self.pipeline.add(nvvidconv)
            
            # Add capsfilter for NVMM memory if available
            if 'caps_filter2' in locals():
                self.pipeline.add(caps_filter2)
        
        self.pipeline.add(encoder)
        self.pipeline.add(muxer)
        self.pipeline.add(filesink)
        
        # Link elements
        if not Gst.Element.link(self.appsrc, videorate):
            raise RuntimeError("Failed to link appsrc to videorate")
        if not Gst.Element.link(videorate, videoconvert1):
            raise RuntimeError("Failed to link videorate to videoconvert1")
            
        # Different linking based on encoder type
        if is_nvidia:
            # Using NVIDIA hardware acceleration
            if 'caps_filter1' in locals():
                # With NV12 capsfilter
                if not Gst.Element.link(videoconvert1, caps_filter1):
                    raise RuntimeError("Failed to link videoconvert1 to caps_filter1")
                if not Gst.Element.link(caps_filter1, nvvidconv):
                    raise RuntimeError("Failed to link caps_filter1 to nvvidconv")
                    
                if 'caps_filter2' in locals():
                    # With NVMM capsfilter
                    if not Gst.Element.link(nvvidconv, caps_filter2):
                        raise RuntimeError("Failed to link nvvidconv to caps_filter2")
                    if not Gst.Element.link(caps_filter2, encoder):
                        raise RuntimeError("Failed to link caps_filter2 to encoder")
                else:
                    # No NVMM capsfilter
                    if not Gst.Element.link(nvvidconv, encoder):
                        raise RuntimeError("Failed to link nvvidconv to encoder")
            else:
                # No NV12 capsfilter
                if not Gst.Element.link(videoconvert1, nvvidconv):
                    raise RuntimeError("Failed to link videoconvert1 to nvvidconv")
                if not Gst.Element.link(nvvidconv, encoder):
                    raise RuntimeError("Failed to link nvvidconv to encoder")
        else:
            # Using software encoder
            if not Gst.Element.link(videoconvert1, encoder):
                raise RuntimeError("Failed to link videoconvert1 to encoder")
                
        if not Gst.Element.link(encoder, muxer):
            raise RuntimeError("Failed to link encoder to muxer")
        if not Gst.Element.link(muxer, filesink):
            raise RuntimeError("Failed to link muxer to filesink")
        
        # Set up bus to catch errors
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message)
    
    def _on_message(self, bus, message):
        """Handle pipeline messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End of stream")
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.error_message = f"Error: {err} {debug}"
            print(self.error_message)
            
            # Check for negotiation errors and log detailed debug info
            if "not-negotiated" in str(err):
                print("\nDetailed error diagnostics:")
                print("=== Pipeline elements and capabilities ===")
                
                # Get pipeline elements for debugging
                iterator = self.pipeline.iterate_elements()
                item = iterator.next()
                while item[0] == Gst.IteratorResult.OK:
                    element = item[1]
                    print(f"Element: {element.get_name()}, type: {element.get_factory().get_name()}")
                    
                    # Get pads for this element
                    pads_iterator = element.iterate_pads()
                    if pads_iterator:
                        pad_item = pads_iterator.next()
                        while pad_item[0] == Gst.IteratorResult.OK:
                            pad = pad_item[1]
                            caps = pad.get_current_caps()
                            if caps:
                                print(f"  Pad: {pad.get_name()}, caps: {caps.to_string()}")
                            else:
                                print(f"  Pad: {pad.get_name()}, no caps")
                            pad_item = pads_iterator.next()
                    
                    item = iterator.next()
                
                print("\nThis indicates a format negotiation issue between elements.")
                print("Possible solutions:")
                print("1. Try different formats (BGR vs RGB vs I420)")
                print("2. Explicitly add format conversion elements")
                print("3. Use software encoder instead of hardware encoder")
                print("4. Fall back to OpenCV VideoWriter")
            
            # Stop the pipeline
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"Warning: {warn} {debug}")
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                print(f"Pipeline state changed from {Gst.Element.state_get_name(old_state)} to {Gst.Element.state_get_name(new_state)}")
                
                # When reaching PLAYING state, log success
                if new_state == Gst.State.PLAYING:
                    print("GStreamer pipeline is now playing")
                    
        elif t == Gst.MessageType.STREAM_STATUS:
            status, owner = message.parse_stream_status()
            status_type = Gst.StreamStatusType(status)
            print(f"Stream status: {status_type.value_name} from {owner.get_name()}")
    
    def _main_loop(self):
        """Run GLib main loop to process GStreamer events"""
        self.loop = GLib.MainLoop()
        self.loop.run()
    
    def start(self):
        """Start the encoder pipeline"""
        if self.is_running:
            return
        
        # Start main loop in a separate thread
        self.main_loop_thread = threading.Thread(target=self._main_loop)
        self.main_loop_thread.daemon = True
        self.main_loop_thread.start()
        
        # Start the pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start GStreamer pipeline")
        
        self.is_running = True
        print(f"GStreamer encoder started. Output file: {self.output_file}")
        print(f"Codec: {self.codec}, Bitrate: {self.bitrate/1024/1024:.2f} Mbps")
        print(f"Preset: {self.preset}, RC Mode: {self.rc_mode.name}")
        print(f"Temporal AQ: {'Enabled' if self.temporal_aq else 'Disabled'}, Spatial AQ: {self.spatial_aq}")
    
    def feed_frame(self, frame):
        """
        Feed a frame to the encoder
        
        Args:
            frame: OpenCV BGR frame (numpy array)
        
        Returns:
            True if frame was processed, False if pipeline has stopped
        """
        if not self.is_running:
            return False
        
        try:
            # Check if frame has the correct size
            h, w, c = frame.shape
            if w != self.width or h != self.height:
                raise ValueError(f"Frame size mismatch. Expected {self.width}x{self.height}, got {w}x{h}")
            
            # Save a test frame if requested (first frame only)
            if args.save_yuv_test and self.frame_count == 0:
                test_filename = "test_original_frame.png"
                cv2.imwrite(test_filename, frame)
                print(f"Saved test frame to {test_filename}")
            
            # Use BGR frame directly - GStreamer will handle conversion
            data = frame.tobytes()
            
            # Create buffer from frame data
            buf = Gst.Buffer.new_wrapped(data)
            
            # Set timestamp
            duration = 1 / self.fps * Gst.SECOND
            buf.duration = duration
            
            # Set timestamp
            buf.pts = self.frame_count * duration
            buf.dts = buf.pts
            
            # Push buffer to appsrc
            ret = self.appsrc.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                print(f"Error pushing buffer to GStreamer pipeline: {ret}")
                if self.error_message:
                    print(f"GStreamer error: {self.error_message}")
                return False
            
            self.frame_count += 1
            return True
            
        except Exception as e:
            print(f"Error feeding frame to GStreamer encoder: {e}")
            return False
    
    def stop(self):
        """Stop the encoder pipeline"""
        if not self.is_running:
            return
        
        # Send EOS event
        self.appsrc.emit("end-of-stream")
        
        # Wait for EOS to propagate
        timeout = 5  # seconds
        start_time = time.time()
        while self.is_running and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        # Force stop if necessary
        if self.is_running:
            self.pipeline.set_state(Gst.State.NULL)
            self.is_running = False
        
        # Stop main loop
        if self.loop and self.loop.is_running():
            self.loop.quit()
        
        # Wait for main loop thread to exit
        if self.main_loop_thread:
            self.main_loop_thread.join(timeout=2)
        
        print("GStreamer encoder stopped")
        
    def __del__(self):
        """Destructor: ensure resources are freed"""
        self.stop()


def create_gstreamer_encoder(output_file, fps, width, height):
    """Create a GStreamer-based encoder for high quality video output"""
    if not USE_GSTREAMER:
        print("GStreamer not available. Falling back to OpenCV encoder.")
        return None
    
    try:
        # Calculate bitrate if not specified
        if args.bitrate != '0':
            try:
                bitrate_mbps = float(args.bitrate)
            except ValueError:
                bitrate_mbps = 0  # Auto bitrate
        else:
            bitrate_mbps = 0  # Auto bitrate
        
        # Check if NVIDIA hardware encoding is requested and available
        use_nvidia = args.use_gpu_encoder
        if use_nvidia:
            # Try to create a simple test pipeline with nvvidconv to check availability
            try:
                test_pipeline = Gst.Pipeline.new("test-pipeline")
                nvvidconv_test = Gst.ElementFactory.make("nvvidconv", "nvvidconv-test")
                if nvvidconv_test:
                    test_pipeline.add(nvvidconv_test)
                    print("NVIDIA hardware encoding elements available")
                else:
                    print("NVIDIA hardware encoding elements not available, using software encoding")
                    use_nvidia = False
            except Exception as e:
                print(f"Error checking NVIDIA elements: {e}")
                print("Falling back to software encoding")
                use_nvidia = False
        
        # Create encoder
        encoder = GstEncoder(
            output_file=output_file,
            width=width,
            height=height,
            fps=fps,
            codec=args.codec,
            bitrate_mbps=bitrate_mbps,
            use_nvidia=use_nvidia,
            preset=args.gst_preset,
            rc_mode=args.rc_mode,
            temporal_aq=args.temporal_aq,
            spatial_aq=args.spatial_aq
        )
        
        # Start encoder
        encoder.start()
        return encoder
    
    except Exception as e:
        print(f"Error creating GStreamer encoder: {e}")
        print("Falling back to OpenCV encoder.")
        return None

if __name__ == "__main__":
    process_video() 