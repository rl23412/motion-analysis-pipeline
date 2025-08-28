import os
import glob
import numpy as np
import cv2
import pandas as pd
from scipy.io import loadmat
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

class RegionVideoOverlayGenerator:
    def __init__(self, base_dir, analysis_outputs_dir, output_dir):
        """
        Initialize the region video overlay generator.
        
        Args:
            base_dir: Base directory containing vid1-vid30 folders with videos
            analysis_outputs_dir: Directory containing MATLAB analysis outputs
            output_dir: Output directory for generated videos
        """
        self.base_dir = base_dir
        self.analysis_outputs_dir = analysis_outputs_dir
        self.output_dir = output_dir
        self.camera_name = "Camera1"  # Default camera
        
        # Reverse mapping from region files back to vid names
        self.vid_mapping = [
            {"dannce_name": "vid1", "original_region": "DRG"},
            {"dannce_name": "vid2", "original_region": "DRG"},
            {"dannce_name": "vid3", "original_region": "IT"},
            {"dannce_name": "vid4", "original_region": "IT"},
            {"dannce_name": "vid5", "original_region": "IT"},
            {"dannce_name": "vid6", "original_region": "SC"},
            {"dannce_name": "vid7", "original_region": "SC"},
            {"dannce_name": "vid8", "original_region": "SC"},
            {"dannce_name": "vid9", "original_region": "DRG"},
            {"dannce_name": "vid10", "original_region": "DRG"},
            {"dannce_name": "vid11", "original_region": "SC"},
            {"dannce_name": "vid12", "original_region": "SC"},
            {"dannce_name": "vid13", "original_region": "SC"},
            {"dannce_name": "vid14", "original_region": "SNI"},
            {"dannce_name": "vid15", "original_region": "SNI"},
            {"dannce_name": "vid16", "original_region": "SNI"},
            {"dannce_name": "vid17", "original_region": "DRG"},
            # session3(week4) videos
            {"dannce_name": "vid18", "original_region": "week4-DRG"},
            {"dannce_name": "vid19", "original_region": "week4-DRG"},
            {"dannce_name": "vid20", "original_region": "week4-DRG"},
            {"dannce_name": "vid21", "original_region": "week4-SC"},
            {"dannce_name": "vid22", "original_region": "week4-SC"},
            {"dannce_name": "vid23", "original_region": "week4-SC"},
            {"dannce_name": "vid24", "original_region": "week4-SNI"},
            {"dannce_name": "vid25", "original_region": "week4-SNI"},
            {"dannce_name": "vid26", "original_region": "week4-SNI"},
            # session4(TBI) videos
            {"dannce_name": "vid27", "original_region": "week4-TBI"},
            {"dannce_name": "vid28", "original_region": "week4-TBI"},
            {"dannce_name": "vid29", "original_region": "week4-TBI"},
            {"dannce_name": "vid30", "original_region": "week4-TBI"},
        ]
        
        # Create reverse mapping: region -> vid_names
        self.region_to_vids = {}
        region_counters = {}
        
        for mapping in self.vid_mapping:
            region = mapping["original_region"]
            vid_name = mapping["dannce_name"]
            
            if region not in self.region_to_vids:
                self.region_to_vids[region] = []
                region_counters[region] = 0
            
            region_counters[region] += 1
            region_file_name = f"{region}_{region_counters[region]}.mat"
            
            self.region_to_vids[region].append({
                "vid_name": vid_name,
                "region_file": region_file_name,
                "region_index": region_counters[region]
            })
        
        print(f"Mapped {len(self.vid_mapping)} videos across {len(self.region_to_vids)} regions")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        for region in self.region_to_vids.keys():
            region_dir = os.path.join(output_dir, region)
            os.makedirs(region_dir, exist_ok=True)
    
    def load_frame_indices(self):
        """Load frame indices from MATLAB analysis outputs and map region names back to vid names"""
        print("Loading frame indices from MATLAB analysis...")
        
        frame_indices_dir = os.path.join(self.analysis_outputs_dir, 'csv', 'frame_indices_per_video')
        if not os.path.exists(frame_indices_dir):
            raise FileNotFoundError(f"Frame indices directory not found: {frame_indices_dir}")
        
        # Load region mapping
        region_mapping_file = os.path.join(self.analysis_outputs_dir, 'csv', 'region_label_mapping.csv')
        region_mapping = pd.read_csv(region_mapping_file)
        print(f"Loaded region mapping with {len(region_mapping)} regions")
        
        # Create reverse mapping from region filename back to vid name
        region_to_vid_mapping = {}
        region_counters = {}
        
        for mapping in self.vid_mapping:
            region = mapping["original_region"]
            vid_name = mapping["dannce_name"]
            
            if region not in region_counters:
                region_counters[region] = 0
            
            region_counters[region] += 1
            region_filename = f"{region}_{region_counters[region]}"  # e.g., "DRG_1", "DRG_2"
            region_to_vid_mapping[region_filename] = vid_name
        
        print(f"Created region-to-vid mapping: {list(region_to_vid_mapping.items())[:5]}...")
        
        # Load frame indices for each video
        self.video_frame_indices = {}
        frame_files = glob.glob(os.path.join(frame_indices_dir, "*_frame_indices.csv"))
        frame_files.sort()  # Sort to ensure consistent ordering
        
        for frame_file in frame_files:
            filename = os.path.basename(frame_file)
            print(f"Processing frame file: {filename}")
            
            # Extract video index and region-based name from filename
            # Format is like "001_DRG_1.mat_frame_indices.csv"
            # We need to handle the .mat part correctly
            
            # First, remove the .csv extension
            name_without_csv = filename.replace('.csv', '')
            print(f"  Name without .csv: {name_without_csv}")
            
            # Split by underscore
            parts = name_without_csv.split('_')
            print(f"  Filename parts: {parts}")
            
            if len(parts) >= 3:  # At least: "001", "DRG", "1.mat", "frame", "indices"
                try:
                    video_idx = int(parts[0])
                    
                    # Find where "frame" starts - everything before that is the region file name
                    region_file_parts = []
                    for i in range(1, len(parts)):
                        if parts[i] == "frame":  # Stop at "frame_indices"
                            break
                        region_file_parts.append(parts[i])
                    
                    # Join the region file parts and remove .mat extension
                    region_file_name = "_".join(region_file_parts)
                    if region_file_name.endswith('.mat'):
                        region_file_name = region_file_name[:-4]  # Remove .mat
                    
                    print(f"  Extracted region file name: '{region_file_name}'")
                    
                    # Convert region file name to standard format
                    # e.g., "DRG_1" -> we want "DRG" as base region
                    if '_' in region_file_name:
                        region_base = '_'.join(region_file_name.split('_')[:-1])  # Everything except the number
                        region_number = region_file_name.split('_')[-1]  # The number
                    else:
                        region_base = region_file_name
                        region_number = '1'
                    
                    # Convert week4_DRG to week4-DRG format
                    if region_base.startswith('week4_'):
                        region_base = region_base.replace('_', '-', 1)
                    
                    region_name = region_base
                    print(f"  Extracted region base: '{region_name}', number: {region_number}")
                    
                    # Try to map region name to vid name using multiple strategies
                    vid_name = None
                    
                    # Strategy 1: Direct mapping using original region file name (e.g., "DRG_1" -> "vid1")
                    region_with_number = f"{region_name}_{region_number}"
                    if region_with_number in region_to_vid_mapping:
                        vid_name = region_to_vid_mapping[region_with_number]
                        print(f"  Direct mapping: {region_with_number} -> {vid_name}")
                    
                    # Strategy 2: Find all videos of this region type and use the number to select
                    elif region_name in [mapping["original_region"] for mapping in self.vid_mapping]:
                        matching_vids = [mapping for mapping in self.vid_mapping if mapping["original_region"] == region_name]
                        region_number_int = int(region_number)
                        if region_number_int <= len(matching_vids):
                            vid_name = matching_vids[region_number_int - 1]["dannce_name"]
                            print(f"  Region sequence mapping: {region_name} #{region_number} -> {vid_name}")
                        else:
                            print(f"  Region number too high for {region_name}: found #{region_number}, expected max #{len(matching_vids)}")
                    
                    # Strategy 3: Use video index directly from the filename
                    elif 1 <= video_idx <= len(self.vid_mapping):
                        vid_name = self.vid_mapping[video_idx - 1]["dannce_name"]
                        print(f"  Index mapping: video_idx {video_idx} -> {vid_name}")
                    
                    # Strategy 4: Failed to map
                    else:
                        print(f"  ERROR: No mapping found for region '{region_name}' #{region_number}")
                        print(f"    Available region types: {set(mapping['original_region'] for mapping in self.vid_mapping)}")
                        print(f"    Available direct mappings: {list(region_to_vid_mapping.keys())}")
                        continue
                    
                    # Load frame indices
                    frame_data = pd.read_csv(frame_file)
                    self.video_frame_indices[vid_name] = {
                        'video_idx': video_idx,
                        'frame_data': frame_data,
                        'original_region_name': region_name,
                        'region_file_name': region_file_name,
                        'region_number': region_number
                    }
                    print(f"  ✅ Loaded frame indices for {vid_name} (from {region_file_name}.mat): {frame_data.shape}")
                    
                except (ValueError, IndexError) as e:
                    print(f"Error processing {filename}: {e}")
                    continue
            else:
                print(f"Unexpected filename format: {filename}")
        
        print(f"\n✅ Successfully loaded frame indices for {len(self.video_frame_indices)} videos")
        print(f"📹 Available video names (mapped back to vid format): {sorted(list(self.video_frame_indices.keys()))}")
        
        # Show a summary of the mapping
        if self.video_frame_indices:
            print(f"\n📊 Mapping Summary:")
            for vid_name, data in sorted(self.video_frame_indices.items()):
                print(f"  {vid_name} ← {data['region_file_name']}.mat (region: {data['original_region_name']})")
        
    
    def load_2d_coordinates_from_file(self, coordinates_2d_file="/work/rl349/coordinates_2D_all_videos.npy"):
        """Load pre-computed 2D coordinates from numpy file"""
        if not hasattr(self, '_coordinates_2d_cache'):
            try:
                print(f"Loading 2D coordinates from: {coordinates_2d_file}")
                self._coordinates_2d_cache = np.load(coordinates_2d_file, allow_pickle=True).item()
                print(f"✅ Loaded 2D coordinates for {len(self._coordinates_2d_cache)} videos")
                
                # Show available video names
                available_videos = sorted(list(self._coordinates_2d_cache.keys()))
                print(f"📹 Available videos in 2D coordinates: {available_videos}")
                
            except FileNotFoundError:
                print(f"❌ Error: 2D coordinates file not found: {coordinates_2d_file}")
                self._coordinates_2d_cache = {}
            except Exception as e:
                print(f"❌ Error loading 2D coordinates: {e}")
                self._coordinates_2d_cache = {}
        
        return self._coordinates_2d_cache
    
    def map_region_filename_to_vid_name(self, region_filename):
        """Map region-based filename back to vid name using the same logic as load_frame_indices"""
        # Create reverse mapping from region filename back to vid name
        region_to_vid_mapping = {}
        region_counters = {}
        
        for mapping in self.vid_mapping:
            region = mapping["original_region"]
            vid_name = mapping["dannce_name"]
            
            if region not in region_counters:
                region_counters[region] = 0
            
            region_counters[region] += 1
            region_filename_key = f"{region}_{region_counters[region]}"  # e.g., "DRG_1", "DRG_2"
            region_to_vid_mapping[region_filename_key] = vid_name
        
        # Remove .mat extension if present
        if region_filename.endswith('.mat'):
            region_filename = region_filename[:-4]
        
        # Try direct mapping first
        if region_filename in region_to_vid_mapping:
            return region_to_vid_mapping[region_filename]
        
        # Try to parse region name and number
        if '_' in region_filename:
            region_base = '_'.join(region_filename.split('_')[:-1])  # Everything except the number
            region_number = region_filename.split('_')[-1]  # The number
        else:
            region_base = region_filename
            region_number = '1'
        
        # Convert week4_DRG to week4-DRG format
        if region_base.startswith('week4_'):
            region_base = region_base.replace('_', '-', 1)
        
        region_name = region_base
        
        # Find all videos of this region type and use the number to select
        if region_name in [mapping["original_region"] for mapping in self.vid_mapping]:
            matching_vids = [mapping for mapping in self.vid_mapping if mapping["original_region"] == region_name]
            region_number_int = int(region_number)
            if region_number_int <= len(matching_vids):
                return matching_vids[region_number_int - 1]["dannce_name"]
        
        print(f"Warning: Could not map region filename '{region_filename}' to vid name")
        return None
    
    def load_2d_coordinates(self, vid_name):
        """Load 2D coordinates for a specific video from pre-computed file"""
        coordinates_2d_all = self.load_2d_coordinates_from_file()
        
        if vid_name in coordinates_2d_all:
            coords_2d = coordinates_2d_all[vid_name]
            print(f"Loaded 2D coordinates for {vid_name}: {coords_2d.shape}")
            return coords_2d
        else:
            print(f"Warning: {vid_name} not found in 2D coordinates file")
            print(f"Available videos: {list(coordinates_2d_all.keys())}")
            return None
    
    def get_video_path(self, vid_name):
        """Get the path to the video file for a given video name"""
        # Try different possible video file names and paths
        possible_paths = [
            os.path.join(self.base_dir, vid_name, "videos", self.camera_name, "0.mp4"),
            os.path.join(self.base_dir, vid_name, "videos", self.camera_name, "0.avi"),
            os.path.join(self.base_dir, vid_name, "videos", self.camera_name, f"{vid_name}.mp4"),
            os.path.join(self.base_dir, vid_name, "videos", self.camera_name, f"{self.camera_name}.mp4"),
        ]
        
        for video_path in possible_paths:
            if os.path.exists(video_path):
                return video_path
        
        # If none found, list what's actually in the directory
        camera_dir = os.path.join(self.base_dir, vid_name, "videos", self.camera_name)
        if os.path.exists(camera_dir):
            files = os.listdir(camera_dir)
            print(f"Warning: Video file not found for {vid_name}")
            print(f"  Searched in: {camera_dir}")
            print(f"  Available files: {files}")
        else:
            video_base_dir = os.path.join(self.base_dir, vid_name, "videos")
            if os.path.exists(video_base_dir):
                cameras = os.listdir(video_base_dir)
                print(f"Warning: Camera directory not found: {camera_dir}")
                print(f"  Available cameras: {cameras}")
            else:
                vid_dir = os.path.join(self.base_dir, vid_name)
                if os.path.exists(vid_dir):
                    contents = os.listdir(vid_dir)
                    print(f"Warning: Videos directory not found: {video_base_dir}")
                    print(f"  {vid_name} directory contains: {contents}")
                else:
                    print(f"Warning: Video directory not found: {vid_dir}")
        
        return None
    
    def compute_centroid(self, coords_2d_frame):
        """Compute centroid of keypoints for a single frame"""
        if coords_2d_frame is None or len(coords_2d_frame) == 0:
            return None
        
        # Debug: Show the shape of the input coordinates
        if hasattr(coords_2d_frame, 'shape'):
            if len(coords_2d_frame.shape) != 2 or coords_2d_frame.shape[1] != 2:
                print(f"    Warning: Unexpected coords_2d_frame shape: {coords_2d_frame.shape}")
                # Try to reshape if possible
                if coords_2d_frame.size % 2 == 0:
                    coords_2d_frame = coords_2d_frame.reshape(-1, 2)
                    print(f"    Reshaped to: {coords_2d_frame.shape}")
                else:
                    print(f"    Cannot reshape coords_2d_frame with size {coords_2d_frame.size}")
                    return None
        
        try:
            # Remove NaN values
            nan_mask = np.isnan(coords_2d_frame)
            if len(nan_mask.shape) > 1:
                # For 2D array, check if any coordinate in a point is NaN
                valid_mask = ~nan_mask.any(axis=1)
            else:
                # For 1D array, just check for NaN values
                valid_mask = ~nan_mask
            
            valid_points = coords_2d_frame[valid_mask]
            if len(valid_points) == 0:
                return None
            
            centroid = np.mean(valid_points, axis=0)
            return centroid
        except Exception as e:
            print(f"    Error computing centroid: {e}")
            print(f"    coords_2d_frame shape: {coords_2d_frame.shape if hasattr(coords_2d_frame, 'shape') else 'no shape'}")
            print(f"    coords_2d_frame type: {type(coords_2d_frame)}")
            return None
    
    def draw_skeleton_on_frame(self, frame, coords_2d_frame, skeleton_color=(0, 255, 255)):
        """
        Draw skeleton connections on frame using mouse14 format.
        
        Mouse14 joint connections (0-indexed):
        [0, 1], [0, 2], [1, 2], [0, 3], [3, 4], [4, 5], [3, 6], [6, 7], 
        [3, 8], [8, 9], [4, 10], [10, 11], [4, 12], [12, 13]
        """
        if coords_2d_frame is None or len(coords_2d_frame) == 0:
            return frame
        
        overlay_frame = frame.copy()
        
        # Mouse14 skeleton connections (0-indexed)
        connections = [
            [0, 1], [0, 2], [1, 2],  # head connections
            [0, 3], [3, 4], [4, 5],  # spine: Snout-SpineF-SpineM-Tail
            [3, 6], [6, 7],          # left front: SpineF-ForShdL-ForepawL  
            [3, 8], [8, 9],          # right front: SpineF-ForeShdR-ForepawR
            [4, 10], [10, 11],       # left hind: SpineM-HindShdL-HindpawL
            [4, 12], [12, 13]        # right hind: SpineM-HindShdR-HindpawR
        ]
        
        try:
            # Ensure coords_2d_frame is the right shape
            if hasattr(coords_2d_frame, 'shape'):
                if len(coords_2d_frame.shape) != 2 or coords_2d_frame.shape[1] != 2:
                    if coords_2d_frame.size % 2 == 0:
                        coords_2d_frame = coords_2d_frame.reshape(-1, 2)
                    else:
                        return frame
            
            # Draw skeleton connections
            for connection in connections:
                joint1_idx, joint2_idx = connection
                if joint1_idx < len(coords_2d_frame) and joint2_idx < len(coords_2d_frame):
                    point1 = coords_2d_frame[joint1_idx]
                    point2 = coords_2d_frame[joint2_idx]
                    
                    # Check if both points are valid
                    if (len(point1) >= 2 and len(point2) >= 2 and 
                        not np.isnan(point1[0]) and not np.isnan(point1[1]) and
                        not np.isnan(point2[0]) and not np.isnan(point2[1])):
                        
                        x1, y1 = int(point1[0]), int(point1[1])
                        x2, y2 = int(point2[0]), int(point2[1])
                        
                        # Check bounds
                        if (0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and
                            0 <= x2 < frame.shape[1] and 0 <= y2 < frame.shape[0]):
                            cv2.line(overlay_frame, (x1, y1), (x2, y2), skeleton_color, 2)
        
        except Exception as e:
            print(f"    Error drawing skeleton: {e}")
            return frame
        
        return overlay_frame
    
    def overlay_keypoints_on_frame(self, frame, coords_2d_frame, centroid=None, 
                                 keypoint_color=(0, 255, 0), centroid_color=(0, 0, 255)):
        """
        Overlay keypoints and skeleton on a video frame.
        """
        if coords_2d_frame is None or len(coords_2d_frame) == 0:
            return frame
        
        # First draw skeleton
        overlay_frame = self.draw_skeleton_on_frame(frame, coords_2d_frame)
        
        try:
            # Ensure coords_2d_frame is the right shape
            if hasattr(coords_2d_frame, 'shape'):
                if len(coords_2d_frame.shape) != 2 or coords_2d_frame.shape[1] != 2:
                    if coords_2d_frame.size % 2 == 0:
                        coords_2d_frame = coords_2d_frame.reshape(-1, 2)
                    else:
                        return overlay_frame
            
            # Draw keypoints as circles
            for i, point in enumerate(coords_2d_frame):
                if len(point) >= 2 and not np.isnan(point[0]) and not np.isnan(point[1]):
                    x, y = int(point[0]), int(point[1])
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        cv2.circle(overlay_frame, (x, y), 3, keypoint_color, -1)
            
            # Draw centroid if provided
            if centroid is not None and len(centroid) >= 2 and not np.isnan(centroid[0]) and not np.isnan(centroid[1]):
                cx, cy = int(centroid[0]), int(centroid[1])
                if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
                    cv2.circle(overlay_frame, (cx, cy), 6, centroid_color, -1)
        
        except Exception as e:
            print(f"    Error overlaying keypoints: {e}")
            return overlay_frame
        
        return overlay_frame
    
    def crop_around_centroid(self, frame, centroid, crop_size=400):
        """
        Crop frame around centroid.
        Similar to keypoint_moseq approach.
        """
        if centroid is None or np.isnan(centroid[0]) or np.isnan(centroid[1]):
            # If no valid centroid, return resized full frame
            return cv2.resize(frame, (crop_size, crop_size))
        
        cx, cy = int(centroid[0]), int(centroid[1])
        half_size = crop_size // 2
        
        # Calculate crop bounds
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(frame.shape[1], cx + half_size)
        y2 = min(frame.shape[0], cy + half_size)
        
        # Crop frame
        cropped = frame[y1:y2, x1:x2]
        
        # Resize to standard size
        if cropped.size > 0:
            resized = cv2.resize(cropped, (crop_size, crop_size))
        else:
            resized = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        
        return resized
    
    def _find_continuous_sequences(self, frame_indices):
        """
        Find continuous sequences of frame indices.
        Returns list of sequences, each sequence is a list of consecutive frame indices.
        """
        if len(frame_indices) == 0:
            return []
        
        # Sort frame indices
        sorted_frames = np.sort(frame_indices)
        sequences = []
        current_sequence = [sorted_frames[0]]
        
        for i in range(1, len(sorted_frames)):
            if sorted_frames[i] == sorted_frames[i-1] + 1:
                # Consecutive frame, add to current sequence
                current_sequence.append(sorted_frames[i])
            else:
                # Gap found, start new sequence
                if len(current_sequence) > 0:
                    sequences.append(current_sequence)
                current_sequence = [sorted_frames[i]]
        
        # Add the last sequence
        if len(current_sequence) > 0:
            sequences.append(current_sequence)
        
        return sequences
    
    def _select_optimal_sequences(self, continuous_sequences, max_frames, min_sequence_length=5):
        """
        Select the optimal (longer) continuous sequences up to max_frames total.
        Prioritizes longer sequences over shorter ones.
        
        Args:
            continuous_sequences: List of sequences from _find_continuous_sequences
            max_frames: Maximum total frames to select
            min_sequence_length: Minimum length of sequence to consider
        """
        if not continuous_sequences:
            return np.array([])
        
        # Filter sequences by minimum length and sort by length (descending)
        valid_sequences = [seq for seq in continuous_sequences if len(seq) >= min_sequence_length]
        valid_sequences.sort(key=len, reverse=True)
        
        if not valid_sequences:
            # If no sequences meet min length, use the longest available sequences
            valid_sequences = sorted(continuous_sequences, key=len, reverse=True)
        
        # Select sequences until we reach max_frames
        selected_frames = []
        total_frames = 0
        
        for sequence in valid_sequences:
            if total_frames + len(sequence) <= max_frames:
                # Can add entire sequence
                selected_frames.extend(sequence)
                total_frames += len(sequence)
                print(f"    Selected sequence of {len(sequence)} frames (total: {total_frames})")
            else:
                # Add partial sequence to reach max_frames
                remaining = max_frames - total_frames
                if remaining > 0:
                    # Take frames from the middle of the sequence for better representation
                    start_idx = len(sequence) // 2 - remaining // 2
                    start_idx = max(0, start_idx)
                    end_idx = start_idx + remaining
                    partial_sequence = sequence[start_idx:end_idx]
                    selected_frames.extend(partial_sequence)
                    total_frames += len(partial_sequence)
                    print(f"    Selected partial sequence of {len(partial_sequence)} frames from {len(sequence)}-frame sequence (total: {total_frames})")
                break
        
        return np.array(sorted(selected_frames))
    
    def create_region_overlay_video(self, region, map_region_idx, max_frames_per_video=500, min_sequence_length=10):
        """
        Create overlay videos for a specific region showing behavior from specific map regions.
        
        Args:
            region: Region name (e.g., 'DRG', 'SC', etc.)
            map_region_idx: Which map region to extract frames from (1-based indexing)
            max_frames_per_video: Maximum number of frames to include per video
        """
        print(f"\n=== Creating overlay video for region {region}, map region {map_region_idx} ===")
        
        if region not in self.region_to_vids:
            print(f"Warning: Region {region} not found in mapping")
            return
        
        # Collect frames from all videos in this region
        all_frames_data = []
        
        for vid_info in self.region_to_vids[region]:
            vid_name = vid_info["vid_name"]
            print(f"Processing {vid_name}...")
            
            # Get video path and coordinates
            video_path = self.get_video_path(vid_name)
            if video_path is None:
                continue
                
            coords_2d = self.load_2d_coordinates(vid_name)
            if coords_2d is None:
                continue
            
            # Get frame indices for this specific map region
            if vid_name not in self.video_frame_indices:
                print(f"Warning: No frame indices found for {vid_name}")
                continue
            
            frame_data = self.video_frame_indices[vid_name]['frame_data']
            region_col = f'Region_{map_region_idx}'
            
            if region_col not in frame_data.columns:
                print(f"Warning: Region column {region_col} not found")
                continue
            
            # Get frame indices for this region (remove NaN values)
            region_frames = frame_data[region_col].dropna().astype(int).values
            
            if len(region_frames) == 0:
                print(f"No frames found for {vid_name} in region {map_region_idx}")
                continue
            
            # Find continuous sequences and select the longer ones
            continuous_sequences = self._find_continuous_sequences(region_frames)
            sequence_info = [(len(seq), seq[0], seq[-1]) for seq in continuous_sequences]
            sequence_info.sort(reverse=True)  # Sort by length descending
            
            print(f"  Found {len(continuous_sequences)} sequences: {sequence_info[:5]}")  # Show top 5
            selected_frames = self._select_optimal_sequences(continuous_sequences, max_frames_per_video, min_sequence_length)
            region_frames = selected_frames
            
            print(f"  Selected {len(region_frames)} optimal frames for {vid_name} in map region {map_region_idx}")
            
            # Store frame data for processing
            all_frames_data.append({
                'vid_name': vid_name,
                'video_path': video_path,
                'coords_2d': coords_2d,
                'frame_indices': region_frames
            })
        
        if not all_frames_data:
            print(f"No valid frame data found for region {region}")
            return
        
        # Create output video
        output_path = os.path.join(self.output_dir, region, f"{region}_map_region_{map_region_idx}_overlay.mp4")
        self._create_compiled_overlay_video(all_frames_data, output_path, region, map_region_idx)
    
    def _create_compiled_overlay_video(self, all_frames_data, output_path, region, map_region_idx):
        """Create a compiled video showing frames from multiple videos with overlays"""
        print(f"Creating compiled overlay video: {output_path}")
        
        # Video settings
        frame_size = 400
        fps = 10  # Slower for better visualization
        
        # Calculate total frames
        total_frames = sum(len(data['frame_indices']) for data in all_frames_data)
        print(f"Total frames to process: {total_frames}")
        
        if total_frames == 0:
            print("No frames to process")
            return
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_size, frame_size))
        
        if not out.isOpened():
            print(f"Error: Could not create output video {output_path}")
            return
        
        frame_count = 0
        
        try:
            for data_idx, frame_data in enumerate(all_frames_data):
                vid_name = frame_data['vid_name']
                video_path = frame_data['video_path']
                coords_2d = frame_data['coords_2d']
                frame_indices = frame_data['frame_indices']
                
                print(f"Processing video {data_idx + 1}/{len(all_frames_data)}: {vid_name}")
                
                # Open video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Warning: Could not open video {video_path}")
                    continue
                
                # Process each frame
                for frame_idx in frame_indices:
                    # Seek to specific frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        print(f"Warning: Could not read frame {frame_idx} from {vid_name}")
                        continue
                    
                    # Get 2D coordinates for this frame
                    if frame_idx < len(coords_2d):
                        coords_2d_frame = coords_2d[frame_idx]
                        centroid = self.compute_centroid(coords_2d_frame)
                        
                        # Overlay keypoints
                        overlay_frame = self.overlay_keypoints_on_frame(frame, coords_2d_frame, centroid)
                        
                        # Crop around centroid
                        cropped_frame = self.crop_around_centroid(overlay_frame, centroid, frame_size)
                        
                        # Add text overlay with video info
                        cv2.putText(cropped_frame, f"{vid_name} - Frame {frame_idx}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(cropped_frame, f"Region {region} - Map {map_region_idx}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Write frame
                        out.write(cropped_frame)
                        frame_count += 1
                    
                    if frame_count % 100 == 0:
                        print(f"Processed {frame_count}/{total_frames} frames")
                
                cap.release()
        
        except Exception as e:
            print(f"Error creating video: {e}")
        
        finally:
            out.release()
        
        print(f"Created overlay video with {frame_count} frames: {output_path}")
    
    def create_grid_video_for_region(self, region, map_region_idx, grid_rows=3, grid_cols=3, max_frames=200, min_sequence_length=10):
        """
        Create a grid video showing multiple videos side by side for a specific region.
        Similar to keypoint_moseq grid movies.
        """
        print(f"\n=== Creating grid video for region {region}, map region {map_region_idx} ===")
        
        if region not in self.region_to_vids:
            print(f"Warning: Region {region} not found in mapping")
            return
        
        videos_in_region = self.region_to_vids[region]
        num_videos = min(len(videos_in_region), grid_rows * grid_cols)
        
        if num_videos == 0:
            print(f"No videos found for region {region}")
            return
        
        print(f"Creating grid video with {num_videos} videos ({grid_rows}x{grid_cols})")
        
        # Prepare video data
        video_data = []
        for i in range(num_videos):
            vid_info = videos_in_region[i]
            vid_name = vid_info["vid_name"]
            
            video_path = self.get_video_path(vid_name)
            coords_2d = self.load_2d_coordinates(vid_name)
            
            if video_path is None or coords_2d is None:
                continue
            
            # Get frame indices for this map region
            if vid_name in self.video_frame_indices:
                frame_data = self.video_frame_indices[vid_name]['frame_data']
                region_col = f'Region_{map_region_idx}'
                
                if region_col in frame_data.columns:
                    region_frames = frame_data[region_col].dropna().astype(int).values
                    
                    # Find continuous sequences and select the longer ones
                    continuous_sequences = self._find_continuous_sequences(region_frames)
                    sequence_info = [(len(seq), seq[0], seq[-1]) for seq in continuous_sequences]
                    sequence_info.sort(reverse=True)  # Sort by length descending
                    
                    print(f"    {vid_name}: Found {len(continuous_sequences)} sequences: {sequence_info[:3]}")
                    region_frames = self._select_optimal_sequences(continuous_sequences, max_frames, min_sequence_length)
                    print(f"    {vid_name}: Selected {len(region_frames)} optimal frames")
                    
                    video_data.append({
                        'vid_name': vid_name,
                        'video_path': video_path,
                        'coords_2d': coords_2d,
                        'frame_indices': region_frames
                    })
        
        if not video_data:
            print(f"No valid video data for grid video")
            return
        
        # Create grid video
        output_path = os.path.join(self.output_dir, region, f"{region}_map_region_{map_region_idx}_grid.mp4")
        self._create_grid_overlay_video(video_data, output_path, region, map_region_idx, 
                                       grid_rows, grid_cols, max_frames)
    
    def _create_grid_overlay_video(self, video_data, output_path, region, map_region_idx, 
                                  grid_rows, grid_cols, max_frames):
        """Create a grid video with multiple videos side by side"""
        print(f"Creating grid overlay video: {output_path}")
        
        cell_size = 200
        grid_width = grid_cols * cell_size
        grid_height = grid_rows * cell_size
        fps = 15
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video {output_path}")
            return
        
        # Open all video captures
        video_captures = []
        for data in video_data:
            cap = cv2.VideoCapture(data['video_path'])
            if cap.isOpened():
                video_captures.append((cap, data))
            else:
                print(f"Warning: Could not open {data['video_path']}")
        
        try:
            # Find minimum number of frames across all videos
            min_frames = min(len(data['frame_indices']) for _, data in video_captures)
            min_frames = min(min_frames, max_frames)
            
            print(f"Creating grid video with {min_frames} frames")
            
            for frame_idx in range(min_frames):
                grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                
                for cell_idx, (cap, data) in enumerate(video_captures):
                    if cell_idx >= grid_rows * grid_cols:
                        break
                    
                    row = cell_idx // grid_cols
                    col = cell_idx % grid_cols
                    
                    # Get the actual frame index for this video
                    if frame_idx < len(data['frame_indices']):
                        actual_frame_idx = data['frame_indices'][frame_idx]
                        
                        # Read frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
                        ret, frame = cap.read()
                        
                        if ret and actual_frame_idx < len(data['coords_2d']):
                            # Get coordinates and overlay
                            coords_2d_frame = data['coords_2d'][actual_frame_idx]
                            centroid = self.compute_centroid(coords_2d_frame)
                            
                            overlay_frame = self.overlay_keypoints_on_frame(frame, coords_2d_frame, centroid)
                            cropped_frame = self.crop_around_centroid(overlay_frame, centroid, cell_size)
                            
                            # Add video label
                            cv2.putText(cropped_frame, data['vid_name'], 
                                       (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            # Place in grid
                            y_start = row * cell_size
                            y_end = (row + 1) * cell_size
                            x_start = col * cell_size
                            x_end = (col + 1) * cell_size
                            
                            grid_frame[y_start:y_end, x_start:x_end] = cropped_frame
                
                # Add title to grid
                title_text = f"{region} - Map Region {map_region_idx} - Frame {frame_idx+1}/{min_frames}"
                cv2.putText(grid_frame, title_text, (10, grid_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(grid_frame)
                
                if (frame_idx + 1) % 50 == 0:
                    print(f"Processed {frame_idx + 1}/{min_frames} frames")
        
        except Exception as e:
            print(f"Error creating grid video: {e}")
        
        finally:
            # Release all resources
            for cap, _ in video_captures:
                cap.release()
            out.release()
        
        print(f"Created grid video: {output_path}")
    
    def create_watershed_region_overlay_video(self, watershed_region_idx, max_frames_total=1000, min_sequence_length=15):
        """
        Create overlay video using quality-ranked sequences from enhanced MATLAB analysis.
        NEW APPROACH: Use pre-computed quality rankings for most representative sequences.
        """
        print(f"\n=== Creating overlay video for Watershed Region {watershed_region_idx} ===")
        
        # Step 1: Load quality-ranked sequences from MATLAB analysis
        print(f"Step 1: Loading quality-ranked sequences for region {watershed_region_idx}...")
        
        sequences_df = self.load_quality_ranked_sequences(watershed_region_idx)
        
        if sequences_df is None or len(sequences_df) == 0:
            print(f"❌ No quality sequences found for region {watershed_region_idx}")
            return
        
        # Step 2: Select sequences up to max_frames_total using quality ranking
        print(f"Step 2: Selecting sequences up to {max_frames_total} total frames...")
        
        selected_sequences = []
        total_frames = 0
        
        for idx, row in sequences_df.iterrows():
            # Get video info and map to vid name
            video_name_from_file = row['VideoName']
            vid_name = self.map_region_filename_to_vid_name(video_name_from_file)
            
            if vid_name is None:
                print(f"   ⚠️  Could not map {video_name_from_file} to vid name")
                continue
                
            coords_2d = self.load_2d_coordinates(vid_name)
            video_path = self.get_video_path(vid_name)
            
            if coords_2d is None or video_path is None:
                continue
            
            # Parse frame indices - MATLAB saves arrays in specific format
            frames_list = []
            try:
                # Strategy 1: Use StartFrame and EndFrame (most reliable from MATLAB)
                if 'StartFrame' in row.index and 'EndFrame' in row.index:
                    start_frame = int(row['StartFrame'])
                    end_frame = int(row['EndFrame'])
                    frames_list = list(range(start_frame, end_frame + 1))
                    print(f"   📋 Using StartFrame-EndFrame: {start_frame}-{end_frame} ({len(frames_list)} frames)")
                
                # Strategy 2: Try to parse Frames column if available
                elif 'Frames' in row.index:
                    frames_str = str(row['Frames'])
                    print(f"   🔍 Raw Frames column: '{frames_str[:100]}...'")  # Show first 100 chars
                    
                    # Handle MATLAB array format
                    if frames_str.startswith('[') and frames_str.endswith(']'):
                        # MATLAB array format: [1 2 3 4 5] or [1;2;3;4;5]
                        inner_str = frames_str[1:-1].replace(';', ' ')
                        frames_list = [int(x) for x in inner_str.split() if x.strip()]
                    elif ',' in frames_str:
                        # Comma-separated format
                        frames_list = [int(x.strip()) for x in frames_str.split(',') if x.strip()]
                    elif frames_str.replace(' ', '').replace('\n', '').isdigit():
                        # Single frame
                        frames_list = [int(frames_str)]
                    else:
                        # Try space-separated
                        frames_list = [int(x) for x in frames_str.split() if x.strip() and x.strip().isdigit()]
                    
                    if frames_list:
                        print(f"   📋 Parsed {len(frames_list)} frames from Frames column")
                    else:
                        print(f"   ⚠️  Could not parse Frames column: '{frames_str}'")
                
                # Strategy 3: If no frames found, skip this sequence
                else:
                    print(f"   ⚠️  No frame data found in row {idx}")
                    print(f"       Available columns: {list(row.index)}")
                    continue
                
                if len(frames_list) < min_sequence_length:
                    continue
                
                # Check if we can fit this sequence
                if total_frames + len(frames_list) <= max_frames_total:
                    # Add full sequence
                    selected_sequences.append({
                        'vid_name': vid_name,
                        'video_path': video_path,
                        'coords_2d': coords_2d,
                        'frames': frames_list,
                        'length': len(frames_list),
                        'quality': row.get('Quality', 0.5),
                        'avg_density': row.get('AvgDensity', 0.5)
                    })
                    total_frames += len(frames_list)
                elif total_frames < max_frames_total:
                    # Add partial sequence to reach max_frames_total
                    remaining = max_frames_total - total_frames
                    partial_frames = frames_list[:remaining]
                    selected_sequences.append({
                        'vid_name': vid_name,
                        'video_path': video_path,
                        'coords_2d': coords_2d,
                        'frames': partial_frames,
                        'length': len(partial_frames),
                        'quality': row.get('Quality', 0.5),
                        'avg_density': row.get('AvgDensity', 0.5)
                    })
                    total_frames += len(partial_frames)
                    break
                    
            except Exception as e:
                print(f"   ⚠️  Error parsing sequence {idx}: {e}")
                continue
        
        print(f"Selected {len(selected_sequences)} quality-ranked sequences with {total_frames} total frames")
        
        # Show top selected sequences
        print(f"Top selected sequences:")
        for i, seq in enumerate(selected_sequences[:5]):
            print(f"  {i+1}. {seq['vid_name']}: {seq['length']} frames, quality={seq['quality']:.3f}")
        
        # Step 3: Create video from quality-selected sequences
        output_dir = os.path.join(self.output_dir, "watershed_regions")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"watershed_region_{watershed_region_idx}_overlay_quality.mp4")
        
        self._create_watershed_overlay_video_from_sequences(selected_sequences, output_path, watershed_region_idx)
    
    def _create_watershed_overlay_video_from_sequences(self, selected_sequences, output_path, watershed_region_idx):
        """Create video from globally selected sequences"""
        print(f"Creating watershed overlay video from {len(selected_sequences)} globally selected sequences: {output_path}")
        
        # Video settings
        frame_size = 400
        fps = 15
        
        # Calculate total frames
        total_frames = sum(seq['length'] for seq in selected_sequences)
        print(f"Total frames to process: {total_frames}")
        
        if total_frames == 0:
            print("No frames to process")
            return
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_size, frame_size))
        
        if not out.isOpened():
            print(f"Error: Could not create output video {output_path}")
            return
        
        frame_count = 0
        
        try:
            for seq_idx, sequence in enumerate(selected_sequences):
                vid_name = sequence['vid_name']
                video_path = sequence['video_path']
                coords_2d = sequence['coords_2d']
                frame_indices = sequence['frames']
                
                print(f"Processing sequence {seq_idx + 1}/{len(selected_sequences)}: {vid_name} ({len(frame_indices)} frames)")
                
                # Open video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Warning: Could not open video {video_path}")
                    continue
                
                # Process each frame in this sequence
                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        continue
                    
                    # Get 2D coordinates for this frame
                    if frame_idx < len(coords_2d):
                        coords_2d_frame = coords_2d[frame_idx]
                        centroid = self.compute_centroid(coords_2d_frame)
                        
                        # Overlay keypoints
                        overlay_frame = self.overlay_keypoints_on_frame(frame, coords_2d_frame, centroid)
                        cropped_frame = self.crop_around_centroid(overlay_frame, centroid, frame_size)
                        
                        # Add text overlay
                        cv2.putText(cropped_frame, f"{vid_name} - Frame {frame_idx}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(cropped_frame, f"Watershed Region {watershed_region_idx}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(cropped_frame, f"Seq {seq_idx+1}/{len(selected_sequences)}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        out.write(cropped_frame)
                        frame_count += 1
                    
                    if frame_count % 100 == 0:
                        print(f"Processed {frame_count}/{total_frames} frames")
                
                cap.release()
        
        except Exception as e:
            print(f"Error creating video: {e}")
        
        finally:
            out.release()
        
        print(f"Created watershed overlay video with {frame_count} frames: {output_path}")
    
    def load_quality_ranked_sequences(self, watershed_region_idx):
        """Load quality-ranked sequences from enhanced MATLAB analysis"""
        try:
            # Try to load per-region top-35 sequences first
            csv_file = os.path.join(self.analysis_outputs_dir, 'csv', f'top_35_sequences_region_{watershed_region_idx}.csv')
            if os.path.exists(csv_file):
                sequences_df = pd.read_csv(csv_file)
                print(f"✅ Loaded {len(sequences_df)} quality-ranked sequences for region {watershed_region_idx}")
                print(f"📊 Available columns: {list(sequences_df.columns)}")
                if len(sequences_df) > 0:
                    print(f"📋 Sample row: {sequences_df.iloc[0].to_dict()}")
                return sequences_df
            else:
                print(f"⚠️  Per-region file not found: {csv_file}")
                
                # Fallback: load global database and filter
                global_file = os.path.join(self.analysis_outputs_dir, 'csv', 'global_sequence_database.csv')
                if os.path.exists(global_file):
                    global_df = pd.read_csv(global_file)
                    region_sequences = global_df[global_df['RegionIndex'] == watershed_region_idx]
                    top_35 = region_sequences.head(35)
                    print(f"✅ Loaded {len(top_35)} sequences from global database for region {watershed_region_idx}")
                    return top_35
                else:
                    print(f"❌ Quality-ranked sequence files not found")
                    print(f"   Searched for: {csv_file}")
                    print(f"   And: {global_file}")
                    print(f"💡 Using basic frame indices instead...")
                    return self._create_fallback_sequences_from_frame_indices(watershed_region_idx)
                    
        except Exception as e:
            print(f"❌ Error loading quality sequences: {e}")
            return self._create_fallback_sequences_from_frame_indices(watershed_region_idx)
    
    def _create_fallback_sequences_from_frame_indices(self, watershed_region_idx):
        """Create fallback sequences using basic frame indices when quality CSV is not available"""
        print(f"Creating fallback sequences for region {watershed_region_idx} from frame indices...")
        
        region_col = f'Region_{watershed_region_idx}'
        fallback_sequences = []
        
        for vid_name, vid_data in self.video_frame_indices.items():
            frame_data = vid_data['frame_data']
            
            if region_col not in frame_data.columns:
                continue
            
            region_frames = frame_data[region_col].dropna().astype(int).values
            if len(region_frames) == 0:
                continue
            
            # Find continuous sequences
            continuous_sequences = self._find_continuous_sequences(region_frames)
            
            # Convert to dataframe format similar to MATLAB output
            for seq_idx, sequence in enumerate(continuous_sequences):
                if len(sequence) >= 10:  # Minimum sequence length
                    fallback_data = {
                        'VideoName': vid_data['region_file_name'],  # Use original region filename for consistency
                        'RegionIndex': watershed_region_idx,
                        'StartFrame': sequence[0],
                        'EndFrame': sequence[-1],
                        'Length': len(sequence),
                        'Quality': len(sequence) / 1000.0,  # Simple quality based on length
                        'AvgDensity': 0.5,  # Default density
                        'MotionVariance': 0.1  # Default motion variance
                    }
                    fallback_sequences.append(fallback_data)
        
        if fallback_sequences:
            # Sort by length (descending) and take top 35
            fallback_sequences.sort(key=lambda x: x['Length'], reverse=True)
            fallback_sequences = fallback_sequences[:35]
            
            # Convert to DataFrame
            fallback_df = pd.DataFrame(fallback_sequences)
            print(f"✅ Created {len(fallback_df)} fallback sequences for region {watershed_region_idx}")
            return fallback_df
        
        return None
    
    def create_watershed_region_grid_video(self, watershed_region_idx, grid_rows=5, grid_cols=7, max_frames=500, min_sequence_length=10):
        """
        Create 5x7 grid video using quality-ranked sequences from enhanced MATLAB analysis.
        NEW APPROACH: Use pre-computed quality rankings for most representative sequences.
        """
        print(f"\n=== Creating 5x7 grid video for Watershed Region {watershed_region_idx} ===")
        
        max_sequences = grid_rows * grid_cols  # 35 sequences for 5x7 grid
        
        # Step 1: Load quality-ranked sequences from MATLAB analysis
        print(f"Step 1: Loading quality-ranked sequences for region {watershed_region_idx}...")
        
        sequences_df = self.load_quality_ranked_sequences(watershed_region_idx)
        
        if sequences_df is None or len(sequences_df) == 0:
            print(f"❌ No quality sequences found for region {watershed_region_idx}")
            print("   Make sure you've run the enhanced analyze_saved_maps_and_counts.m script first")
            return
        
        # Step 2: Convert dataframe to sequence objects and validate
        print(f"Step 2: Processing {len(sequences_df)} quality-ranked sequences...")
        
        selected_sequences = []
        
        for idx, row in sequences_df.iterrows():
            # Map video name back to get coordinates
            video_name_from_file = row['VideoName']
            vid_name = self.map_region_filename_to_vid_name(video_name_from_file)
            
            if vid_name is None:
                print(f"   ⚠️  Could not map {video_name_from_file} to vid name")
                continue
            
            # Find corresponding video data
            coords_2d = self.load_2d_coordinates(vid_name)
            video_path = self.get_video_path(vid_name)
            
            if coords_2d is None or video_path is None:
                print(f"   ⚠️  Skipping {video_name_from_file}: missing coordinates or video")
                continue
            
            # Parse frame indices from the row (handle MATLAB CSV format)
            frames_list = []
            try:
                # Strategy 1: Use StartFrame and EndFrame (most reliable from MATLAB)
                if 'StartFrame' in row.index and 'EndFrame' in row.index:
                    start_frame = int(row['StartFrame'])
                    end_frame = int(row['EndFrame'])
                    frames_list = list(range(start_frame, end_frame + 1))
                    print(f"   📋 Using StartFrame-EndFrame: {start_frame}-{end_frame} ({len(frames_list)} frames)")
                
                # Strategy 2: Try to parse Frames column if available
                elif 'Frames' in row.index:
                    frames_str = str(row['Frames'])
                    print(f"   🔍 Raw Frames column: '{frames_str[:100]}...'")  # Show first 100 chars
                    
                    # Handle MATLAB array format
                    if frames_str.startswith('[') and frames_str.endswith(']'):
                        # MATLAB array format: [1 2 3 4 5] or [1;2;3;4;5]
                        inner_str = frames_str[1:-1].replace(';', ' ')
                        frames_list = [int(x) for x in inner_str.split() if x.strip()]
                    elif ',' in frames_str:
                        # Comma-separated format
                        frames_list = [int(x.strip()) for x in frames_str.split(',') if x.strip()]
                    elif frames_str.replace(' ', '').replace('\n', '').isdigit():
                        # Single frame
                        frames_list = [int(frames_str)]
                    else:
                        # Try space-separated
                        frames_list = [int(x) for x in frames_str.split() if x.strip() and x.strip().isdigit()]
                    
                    if frames_list:
                        print(f"   📋 Parsed {len(frames_list)} frames from Frames column")
                    else:
                        print(f"   ⚠️  Could not parse Frames column: '{frames_str}'")
                
                # Strategy 3: If no frames found, skip this sequence
                else:
                    print(f"   ⚠️  No frame data found in row {idx}")
                    print(f"       Available columns: {list(row.index)}")
                    continue
                
                if len(frames_list) < min_sequence_length:
                    continue
                
                selected_sequences.append({
                    'vid_name': vid_name,
                    'video_path': video_path,
                    'coords_2d': coords_2d,
                    'frames': frames_list,
                    'length': len(frames_list),
                    'start_frame': frames_list[0],
                    'end_frame': frames_list[-1],
                    'quality': row.get('Quality', 0.5),
                    'avg_density': row.get('AvgDensity', 0.5),
                    'motion_variance': row.get('MotionVariance', 0.1)
                })
                
            except Exception as e:
                print(f"   ⚠️  Error parsing frames for sequence {idx}: {e}")
                continue
            
            # Stop when we have enough sequences
            if len(selected_sequences) >= max_sequences:
                break
        
        print(f"Successfully processed {len(selected_sequences)} quality-ranked sequences")
        
        if not selected_sequences:
            print(f"❌ No valid sequences after processing for region {watershed_region_idx}")
            return
        
        # Show quality metrics of selected sequences
        print(f"Selected sequences (top {len(selected_sequences)}):")
        for i, seq in enumerate(selected_sequences[:5]):  # Show top 5
            print(f"  {i+1}. {seq['vid_name']}: {seq['length']} frames, quality={seq['quality']:.3f}")
        if len(selected_sequences) > 5:
            print(f"  ... and {len(selected_sequences)-5} more sequences")
        
        # Step 3: Create 5x7 grid video from quality-selected sequences
        output_dir = os.path.join(self.output_dir, "watershed_regions")
        output_path = os.path.join(output_dir, f"watershed_region_{watershed_region_idx}_grid_quality.mp4")
        
        self._create_watershed_grid_video_from_sequences(selected_sequences, output_path, watershed_region_idx, grid_rows, grid_cols)
    
    def _create_watershed_grid_video_from_sequences(self, selected_sequences, output_path, watershed_region_idx, grid_rows, grid_cols):
        """Create a 5x7 grid video from globally selected sequences"""
        print(f"Creating 5x7 grid video from {len(selected_sequences)} sequences: {output_path}")
        
        cell_size = 200
        grid_width = grid_cols * cell_size   # 7 * 200 = 1400
        grid_height = grid_rows * cell_size  # 5 * 200 = 1000
        fps = 15
        
        # Calculate frames per sequence for synchronized playback
        if not selected_sequences:
            print("No sequences to process")
            return
        
        # Use the minimum sequence length to ensure all cells have content
        min_seq_length = min(seq['length'] for seq in selected_sequences)
        max_frames_to_use = min(500, min_seq_length)  # Use 500 as default max frames
        
        print(f"Grid size: {grid_width}x{grid_height}, using {max_frames_to_use} frames per sequence")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video {output_path}")
            return
        
        # Open video captures for all sequences
        video_captures = []
        for seq in selected_sequences:
            cap = cv2.VideoCapture(seq['video_path'])
            if cap.isOpened():
                video_captures.append((cap, seq))
            else:
                print(f"Warning: Could not open {seq['video_path']}")
        
        try:
            print(f"Processing {max_frames_to_use} frames for grid video...")
            
            for frame_idx in range(max_frames_to_use):
                grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                
                for cell_idx, (cap, seq_data) in enumerate(video_captures):
                    if cell_idx >= grid_rows * grid_cols:
                        break
                    
                    row = cell_idx // grid_cols
                    col = cell_idx % grid_cols
                    
                    # Get the actual frame index for this sequence
                    if frame_idx < len(seq_data['frames']):
                        actual_frame_idx = seq_data['frames'][frame_idx]
                        
                        # Read frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
                        ret, frame = cap.read()
                        
                        if ret and actual_frame_idx < len(seq_data['coords_2d']):
                            try:
                                # Get coordinates and overlay skeleton + keypoints
                                coords_2d_frame = seq_data['coords_2d'][actual_frame_idx]
                                centroid = self.compute_centroid(coords_2d_frame)
                                
                                overlay_frame = self.overlay_keypoints_on_frame(frame, coords_2d_frame, centroid)
                                cropped_frame = self.crop_around_centroid(overlay_frame, centroid, cell_size)
                                
                                # Add video label and sequence info
                                cv2.putText(cropped_frame, f"{seq_data['vid_name']}", 
                                           (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                cv2.putText(cropped_frame, f"#{cell_idx+1}", 
                                           (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                
                                # Place in grid
                                y_start = row * cell_size
                                y_end = (row + 1) * cell_size
                                x_start = col * cell_size
                                x_end = (col + 1) * cell_size
                                
                                grid_frame[y_start:y_end, x_start:x_end] = cropped_frame
                                
                            except Exception as e:
                                # Fill with black frame on error
                                black_frame = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                                cv2.putText(black_frame, f"{seq_data['vid_name']} - Error", 
                                           (5, cell_size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                y_start = row * cell_size
                                y_end = (row + 1) * cell_size
                                x_start = col * cell_size
                                x_end = (col + 1) * cell_size
                                
                                grid_frame[y_start:y_end, x_start:x_end] = black_frame
                
                # Add title to grid
                title_text = f"Watershed Region {watershed_region_idx} - Top 35 Sequences - Frame {frame_idx+1}/{max_frames_to_use}"
                cv2.putText(grid_frame, title_text, (10, grid_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(grid_frame)
                
                if (frame_idx + 1) % 25 == 0:
                    print(f"Processed {frame_idx + 1}/{max_frames_to_use} frames")
        
        except Exception as e:
            print(f"Error creating grid video: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Release all resources
            for cap, _ in video_captures:
                cap.release()
            out.release()
        
        print(f"Created 5x7 grid video with {len(selected_sequences)} sequences: {output_path}")
    
    def _create_watershed_overlay_video(self, all_video_data, output_path, watershed_region_idx):
        """Create a compiled video showing frames from multiple videos for a watershed region"""
        print(f"Creating watershed overlay video: {output_path}")
        
        # Video settings
        frame_size = 400
        fps = 15  # Good speed for analyzing behavior
        
        # Calculate total frames
        total_frames = sum(len(data['frame_indices']) for data in all_video_data)
        print(f"Total frames to process: {total_frames}")
        
        if total_frames == 0:
            print("No frames to process")
            return
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_size, frame_size))
        
        if not out.isOpened():
            print(f"Error: Could not create output video {output_path}")
            return
        
        frame_count = 0
        
        try:
            for data_idx, video_data in enumerate(all_video_data):
                vid_name = video_data['vid_name']
                video_path = video_data['video_path']
                coords_2d = video_data['coords_2d']
                frame_indices = video_data['frame_indices']
                
                print(f"Processing video {data_idx + 1}/{len(all_video_data)}: {vid_name} ({len(frame_indices)} frames)")
                
                # Open video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Warning: Could not open video {video_path}")
                    continue
                
                # Process each frame
                for frame_idx in frame_indices:
                    # Seek to specific frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if not ret:
                        print(f"Warning: Could not read frame {frame_idx} from {vid_name}")
                        continue
                    
                    # Get 2D coordinates for this frame
                    if frame_idx < len(coords_2d):
                        coords_2d_frame = coords_2d[frame_idx]
                        centroid = self.compute_centroid(coords_2d_frame)
                        
                        # Overlay keypoints
                        overlay_frame = self.overlay_keypoints_on_frame(frame, coords_2d_frame, centroid)
                        
                        # Crop around centroid
                        cropped_frame = self.crop_around_centroid(overlay_frame, centroid, frame_size)
                        
                        # Add text overlay with video info
                        cv2.putText(cropped_frame, f"{vid_name} - Frame {frame_idx}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(cropped_frame, f"Watershed Region {watershed_region_idx}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Write frame
                        out.write(cropped_frame)
                        frame_count += 1
                    
                    if frame_count % 50 == 0:
                        print(f"Processed {frame_count}/{total_frames} frames")
                
                cap.release()
        
        except Exception as e:
            print(f"Error creating video: {e}")
        
        finally:
            out.release()
        
        print(f"Created watershed overlay video with {frame_count} frames: {output_path}")
    
    def _create_watershed_grid_video(self, video_data, output_path, watershed_region_idx, grid_rows, grid_cols):
        """Create a grid video with multiple videos side by side for a watershed region"""
        print(f"Creating watershed grid video: {output_path}")
        
        cell_size = 200
        grid_width = grid_cols * cell_size
        grid_height = grid_rows * cell_size
        fps = 15
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (grid_width, grid_height))
        
        if not out.isOpened():
            print(f"Error: Could not create output video {output_path}")
            return
        
        # Open all video captures
        video_captures = []
        for data in video_data:
            cap = cv2.VideoCapture(data['video_path'])
            if cap.isOpened():
                video_captures.append((cap, data))
            else:
                print(f"Warning: Could not open {data['video_path']}")
        
        try:
            # Find minimum number of frames across all videos
            min_frames = min(len(data['frame_indices']) for _, data in video_captures) if video_captures else 0
            
            print(f"Creating watershed grid video with {min_frames} frames")
            
            for frame_idx in range(min_frames):
                grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
                
                for cell_idx, (cap, data) in enumerate(video_captures):
                    if cell_idx >= grid_rows * grid_cols:
                        break
                    
                    row = cell_idx // grid_cols
                    col = cell_idx % grid_cols
                    
                    # Get the actual frame index for this video
                    if frame_idx < len(data['frame_indices']):
                        actual_frame_idx = data['frame_indices'][frame_idx]
                        
                        # Read frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
                        ret, frame = cap.read()
                        
                        if ret and actual_frame_idx < len(data['coords_2d']):
                            try:
                                # Get coordinates and overlay
                                coords_2d_frame = data['coords_2d'][actual_frame_idx]
                                
                                # Debug coordinate shape for first few frames
                                if frame_idx < 3 and cell_idx < 2:
                                    print(f"    Debug: coords_2d_frame shape for {data['vid_name']} frame {actual_frame_idx}: {coords_2d_frame.shape if hasattr(coords_2d_frame, 'shape') else 'no shape'}")
                                
                                centroid = self.compute_centroid(coords_2d_frame)
                                
                                overlay_frame = self.overlay_keypoints_on_frame(frame, coords_2d_frame, centroid)
                                cropped_frame = self.crop_around_centroid(overlay_frame, centroid, cell_size)
                                
                                # Add video label
                                cv2.putText(cropped_frame, data['vid_name'], 
                                           (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                # Place in grid
                                y_start = row * cell_size
                                y_end = (row + 1) * cell_size
                                x_start = col * cell_size
                                x_end = (col + 1) * cell_size
                                
                                grid_frame[y_start:y_end, x_start:x_end] = cropped_frame
                                
                            except Exception as e:
                                print(f"    Error processing frame {actual_frame_idx} for {data['vid_name']}: {e}")
                                # Fill with black frame on error
                                black_frame = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
                                cv2.putText(black_frame, f"{data['vid_name']} - Error", 
                                           (5, cell_size//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                y_start = row * cell_size
                                y_end = (row + 1) * cell_size
                                x_start = col * cell_size
                                x_end = (col + 1) * cell_size
                                
                                grid_frame[y_start:y_end, x_start:x_end] = black_frame
                
                # Add title to grid
                title_text = f"Watershed Region {watershed_region_idx} - Frame {frame_idx+1}/{min_frames}"
                cv2.putText(grid_frame, title_text, (10, grid_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(grid_frame)
                
                if (frame_idx + 1) % 25 == 0:
                    print(f"Processed {frame_idx + 1}/{min_frames} frames")
        
        except Exception as e:
            print(f"Error creating watershed grid video: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Release all resources
            for cap, _ in video_captures:
                cap.release()
            out.release()
        
        print(f"Created watershed grid video: {output_path}")
    
    def generate_all_watershed_region_videos(self, max_watershed_regions=22):
        """Generate overlay videos for each watershed region across all videos"""
        print(f"\n{'='*60}")
        print("GENERATING WATERSHED REGION OVERLAY VIDEOS")
        print("Optimizing for longer continuous behavioral sequences")
        print(f"{'='*60}")
        
        # Load frame indices
        self.load_frame_indices()
        
        # Create main output directory for watershed regions
        watershed_output_dir = os.path.join(self.output_dir, "watershed_regions")
        os.makedirs(watershed_output_dir, exist_ok=True)
        
        # Generate videos for each watershed region
        for watershed_region_idx in range(1, max_watershed_regions + 1):
            print(f"\n{'='*50}")
            print(f"Processing Watershed Region {watershed_region_idx}")
            print(f"{'='*50}")
            
            try:
                # Create overlay video for this watershed region
                self.create_watershed_region_overlay_video(watershed_region_idx)
                
                # Create grid video for this watershed region
                self.create_watershed_region_grid_video(watershed_region_idx)
                
            except Exception as e:
                print(f"Error processing watershed region {watershed_region_idx}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print("WATERSHED REGION VIDEO GENERATION COMPLETE")
        print(f"{'='*60}")
        
    def test_setup(self):
        """Test function to diagnose setup issues"""
        print(f"\n{'='*60}")
        print("TESTING SETUP - DIAGNOSING ISSUES")
        print(f"{'='*60}")
        
        # Test 1: Check if base directory exists and what's in it
        print(f"\n1. Checking base directory: {self.base_dir}")
        if os.path.exists(self.base_dir):
            contents = os.listdir(self.base_dir)
            vid_dirs = [d for d in contents if d.startswith('vid') and os.path.isdir(os.path.join(self.base_dir, d))]
            print(f"   Base directory exists")
            print(f"   Video directories found: {sorted(vid_dirs)[:10]}...")  # Show first 10
        else:
            print(f"   ERROR: Base directory does not exist!")
            return
        
        # Test 2: Check specific video directory structure
        test_vid = "vid1"
        test_vid_path = os.path.join(self.base_dir, test_vid)
        print(f"\n2. Checking {test_vid} directory structure:")
        if os.path.exists(test_vid_path):
            contents = os.listdir(test_vid_path)
            print(f"   {test_vid} contents: {contents}")
            
            # Check videos subdirectory
            videos_path = os.path.join(test_vid_path, "videos")
            if os.path.exists(videos_path):
                cameras = os.listdir(videos_path)
                print(f"   Available cameras: {cameras}")
                
                # Check Camera1 specifically
                camera1_path = os.path.join(videos_path, "Camera1")
                if os.path.exists(camera1_path):
                    files = os.listdir(camera1_path)
                    print(f"   Camera1 files: {files}")
                else:
                    print(f"   Camera1 directory not found")
            else:
                print(f"   Videos directory not found")
        else:
            print(f"   {test_vid} directory not found")
        
        # Test 3: Check analysis outputs directory
        print(f"\n3. Checking analysis outputs: {self.analysis_outputs_dir}")
        if os.path.exists(self.analysis_outputs_dir):
            csv_dir = os.path.join(self.analysis_outputs_dir, 'csv')
            frame_indices_dir = os.path.join(csv_dir, 'frame_indices_per_video')
            
            if os.path.exists(frame_indices_dir):
                frame_files = glob.glob(os.path.join(frame_indices_dir, "*_frame_indices.csv"))
                print(f"   Found {len(frame_files)} frame index files")
                if frame_files:
                    print(f"   Sample files: {[os.path.basename(f) for f in frame_files[:5]]}")
            else:
                print(f"   Frame indices directory not found: {frame_indices_dir}")
        else:
            print(f"   Analysis outputs directory not found")
        
        # Test 4: Check what frame index files we actually have
        print(f"\n4. Checking frame index files in detail:")
        csv_dir = os.path.join(self.analysis_outputs_dir, 'csv')
        frame_indices_dir = os.path.join(csv_dir, 'frame_indices_per_video')
        
        if os.path.exists(frame_indices_dir):
            frame_files = glob.glob(os.path.join(frame_indices_dir, "*_frame_indices.csv"))
            print(f"   Found {len(frame_files)} frame index files:")
            for frame_file in frame_files:
                filename = os.path.basename(frame_file)
                print(f"     - {filename}")
        
        # Test 5: Try to load frame indices and see what we get
        print(f"\n5. Testing frame indices loading with new mapping:")
        try:
            self.load_frame_indices()
        except Exception as e:
            print(f"   Error loading frame indices: {e}")
        
        # Test 6: Try to load 2D coordinates
        print(f"\n6. Testing 2D coordinates loading:")
        try:
            coordinates_2d = self.load_2d_coordinates_from_file()
            if coordinates_2d:
                sample_vid = list(coordinates_2d.keys())[0] if coordinates_2d else None
                if sample_vid:
                    coords_shape = coordinates_2d[sample_vid].shape
                    print(f"   ✅ 2D coordinates loaded successfully")
                    print(f"   📊 Sample: {sample_vid} has shape {coords_shape}")
                else:
                    print(f"   ⚠️  2D coordinates file is empty")
            else:
                print(f"   ❌ Failed to load 2D coordinates")
        except Exception as e:
            print(f"   ❌ Error loading 2D coordinates: {e}")
        
        print(f"\n{'='*60}")
        print("SETUP TESTING COMPLETE")
        print(f"{'='*60}")
    
    def generate_all_region_videos(self, max_map_regions=22):
        """Legacy method - now redirects to watershed region generation"""
        self.generate_all_watershed_region_videos(max_map_regions)


def main():
    """Main function to run the region video overlay generation"""
    
    # Configuration
    base_dir = "/work/rl349/dannce/mouse14/videos"  # Directory containing vid1-vid30 folders
    analysis_outputs_dir = "analysis_outputs"  # MATLAB analysis outputs directory
    output_dir = "region_video_overlays"  # Output directory for generated videos
    
    print("="*60)
    print("REGION VIDEO OVERLAY GENERATOR")
    print("="*60)
    print(f"Base directory: {base_dir}")
    print(f"Analysis outputs: {analysis_outputs_dir}")
    print(f"Output directory: {output_dir}")
    
    # Initialize generator
    generator = RegionVideoOverlayGenerator(
        base_dir=base_dir,
        analysis_outputs_dir=analysis_outputs_dir,
        output_dir=output_dir
    )
    
    # Test setup first to diagnose issues
    generator.test_setup()
    
    # If the test setup worked and we have frame indices, generate videos
    if hasattr(generator, 'video_frame_indices') and len(generator.video_frame_indices) > 0:
        print(f"\n✅ Setup successful! Found {len(generator.video_frame_indices)} videos.")
        print("Proceeding with video generation...")
        
        # Generate all videos (using watershed regions)
        generator.generate_all_watershed_region_videos(max_watershed_regions=22)
    else:
        print(f"\n❌ Setup failed or no frame indices found. Please check the diagnostic output above.")
        print("Video generation skipped.")
    
    print("\nVideo generation complete!")
    print(f"Check output directory: {output_dir}")


if __name__ == "__main__":
    main()
