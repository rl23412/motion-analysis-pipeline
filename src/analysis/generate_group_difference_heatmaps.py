#!/usr/bin/env python3
"""
Group Difference Heatmap Generator

This script generates density heatmaps showing frequency differences between:
- "Other groups as a whole" (DRG, IT, SC, SNI, week4-DRG, week4-SC, week4-SNI)
- "TBI only groups" (week4-TBI)

The heatmaps show spatial patterns of behavioral differences across watershed regions,
with positive values indicating higher frequency in other groups and negative values 
indicating higher frequency in TBI groups.

Author: Generated for behavioral embedding analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.io import loadmat
from pathlib import Path
import warnings
# cv2 not needed for heatmap analysis
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
warnings.filterwarnings('ignore')

class GroupDifferenceHeatmapGenerator:
    """Generate difference heatmaps comparing group frequencies across behavioral maps"""
    
    def __init__(self, analysis_outputs_dir='analysis_outputs'):
        """
        Initialize the heatmap generator
        
        Args:
            analysis_outputs_dir: Directory containing MATLAB analysis outputs
        """
        self.analysis_outputs_dir = Path(analysis_outputs_dir)
        self.csv_dir = self.analysis_outputs_dir / 'csv'
        self.figures_dir = self.analysis_outputs_dir / 'figures'
        self.heatmap_output_dir = self.analysis_outputs_dir / 'group_difference_heatmaps'
        
        # Create output directory
        self.heatmap_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Data storage
        self.region_counts = None
        self.region_frequencies = None
        self.group_frequencies = None
        self.difference_maps = None
        self.watershed_data = None
        
        print("="*60)
        print("GROUP DIFFERENCE HEATMAP GENERATOR")
        print("="*60)
        print(f"Analysis outputs directory: {self.analysis_outputs_dir}")
        print(f"Output directory: {self.heatmap_output_dir}")
    
    def parse_filename_to_group(self, filename):
        """
        Parse filename to extract experimental group with TBI vs Other classification
        
        Args:
            filename: e.g., 'DRG_1.mat', 'week4-TBI_2.mat'
            
        Returns:
            dict with 'group', 'is_tbi', and 'group_type' keys
        """
        # Remove .mat extension if present
        name = filename.replace('.mat', '')
        
        # Check if it's a week4 file
        if name.startswith('week4-'):
            week = 'week4'
            name = name[6:]  # Remove 'week4-' prefix
        else:
            week = 'week1'
        
        # Extract base group
        for group in ['DRG', 'IT', 'SC', 'SNI', 'TBI']:
            if name.startswith(group):
                if week == 'week4':
                    full_group = f'week4-{group}'
                else:
                    full_group = group
                
                # Classify as TBI vs Other
                is_tbi = (group == 'TBI')
                group_type = 'TBI' if is_tbi else 'Other'
                
                return {
                    'group': full_group,
                    'base_group': group,
                    'week': week,
                    'is_tbi': is_tbi,
                    'group_type': group_type
                }
        
        return {
            'group': 'unknown',
            'base_group': 'unknown',
            'week': 'unknown',
            'is_tbi': False,
            'group_type': 'Other'
        }
    
    def load_frequency_data(self):
        """Load and process frequency data from existing analysis outputs"""
        print("\n1. Loading frequency data...")
        
        # Load region counts
        counts_file = self.csv_dir / 'per_file_region_counts.csv'
        if not counts_file.exists():
            raise FileNotFoundError(f"Region counts file not found: {counts_file}")
        
        self.region_counts = pd.read_csv(counts_file)
        print(f"✅ Loaded region counts: {self.region_counts.shape}")
        
        # Load individual video frequencies if available
        freq_file = self.csv_dir / 'individual_video_frequencies.csv'
        if freq_file.exists():
            self.region_frequencies = pd.read_csv(freq_file)
            print(f"✅ Loaded individual frequencies: {self.region_frequencies.shape}")
        else:
            # Calculate frequencies from counts and total frames
            print("📊 Calculating frequencies from counts...")
            self._calculate_frequencies_from_counts()
        
        # Get available regions
        region_columns = [col for col in self.region_counts.columns if col.startswith('Region_')]
        self.available_regions = [int(col.split('_')[1]) for col in region_columns]
        print(f"Available regions: {sorted(self.available_regions)}")
        
        return self.region_frequencies
    
    def _calculate_frequencies_from_counts(self):
        """Calculate frequencies from counts if individual frequencies not available"""
        
        # Load frame summary to get total frames per video
        summary_file = self.csv_dir / 'frame_indices_summary.csv'
        if summary_file.exists():
            summary_data = pd.read_csv(summary_file)
            print(f"✅ Loaded frame summary: {summary_data.shape}")
        else:
            print("⚠️  Frame summary not found, using alternative method")
            # Use sum of all region counts as approximation for total frames
            region_cols = [col for col in self.region_counts.columns if col.startswith('Region_')]
            total_frames = self.region_counts[region_cols].sum(axis=1)
            summary_data = pd.DataFrame({
                'File': self.region_counts['File'],
                'total_frames': total_frames
            })
        
        # Calculate frequencies
        frequency_data = []
        for idx, row in self.region_counts.iterrows():
            filename = row['File']
            
            # Get total frames for this video
            if 'total_frames' in summary_data.columns:
                total_frames = summary_data[summary_data['File'] == filename]['total_frames'].iloc[0] if len(summary_data[summary_data['File'] == filename]) > 0 else 1
            else:
                # Fallback: sum all region counts
                region_cols = [col for col in row.index if col.startswith('Region_')]
                total_frames = row[region_cols].sum()
            
            freq_row = {'filename': filename, 'total_frames': total_frames}
            
            # Calculate frequency for each region
            for col in row.index:
                if col.startswith('Region_'):
                    count = row[col]
                    frequency = count / total_frames if total_frames > 0 else 0
                    freq_col = col.replace('Region_', 'Freq_Region_')
                    freq_row[freq_col] = frequency
                    freq_row[col] = count  # Keep original count
            
            frequency_data.append(freq_row)
        
        self.region_frequencies = pd.DataFrame(frequency_data)
        print(f"✅ Calculated frequencies for {len(self.region_frequencies)} videos")
    
    def group_videos_by_type(self):
        """Group videos into TBI vs Other groups and calculate group statistics"""
        print("\n2. Grouping videos by experimental type...")
        
        grouped_data = []
        
        for idx, row in self.region_frequencies.iterrows():
            filename = row['filename']
            group_info = self.parse_filename_to_group(filename)
            
            # Get frequencies for all regions
            freq_data = {}
            for region in self.available_regions:
                freq_col = f'Freq_Region_{region}'
                if freq_col in self.region_frequencies.columns:
                    freq_data[f'freq_region_{region}'] = row[freq_col]
                else:
                    freq_data[f'freq_region_{region}'] = 0
            
            # Combine group info with frequency data
            video_data = {
                'filename': filename,
                'total_frames': row.get('total_frames', 1),
                **group_info,
                **freq_data
            }
            
            grouped_data.append(video_data)
        
        self.group_frequencies = pd.DataFrame(grouped_data)
        
        # Display group distribution
        group_counts = self.group_frequencies.groupby(['group_type', 'group']).size().reset_index(name='count')
        print("\n📊 Group distribution:")
        print(group_counts.to_string(index=False))
        
        # Calculate group-level statistics
        self._calculate_group_statistics()
        
        return self.group_frequencies
    
    def _calculate_group_statistics(self):
        """Calculate mean frequencies per group type (TBI vs Other)"""
        print("\n3. Calculating group statistics...")
        
        self.group_stats = {}
        
        for group_type in ['TBI', 'Other']:
            group_data = self.group_frequencies[self.group_frequencies['group_type'] == group_type]
            
            if len(group_data) == 0:
                print(f"⚠️  No videos found for group type: {group_type}")
                continue
            
            # Calculate statistics for each region
            region_stats = {}
            for region in self.available_regions:
                freq_col = f'freq_region_{region}'
                if freq_col in group_data.columns:
                    frequencies = group_data[freq_col].values
                    region_stats[region] = {
                        'mean': np.mean(frequencies),
                        'std': np.std(frequencies),
                        'sem': np.std(frequencies) / np.sqrt(len(frequencies)) if len(frequencies) > 0 else 0,
                        'n_videos': len(frequencies),
                        'median': np.median(frequencies)
                    }
            
            self.group_stats[group_type] = region_stats
            print(f"✅ {group_type} group: {len(group_data)} videos, {len(region_stats)} regions")
        
        # Calculate differences (Other - TBI)
        if 'Other' in self.group_stats and 'TBI' in self.group_stats:
            self.difference_stats = {}
            for region in self.available_regions:
                if region in self.group_stats['Other'] and region in self.group_stats['TBI']:
                    other_mean = self.group_stats['Other'][region]['mean']
                    tbi_mean = self.group_stats['TBI'][region]['mean']
                    difference = other_mean - tbi_mean
                    
                    self.difference_stats[region] = {
                        'other_mean': other_mean,
                        'tbi_mean': tbi_mean,
                        'difference': difference,  # Positive = higher in Other groups
                        'other_n': self.group_stats['Other'][region]['n_videos'],
                        'tbi_n': self.group_stats['TBI'][region]['n_videos']
                    }
            
            print(f"✅ Calculated differences for {len(self.difference_stats)} regions")
        else:
            print("❌ Cannot calculate differences - missing group data")
    
    def load_watershed_map_data(self):
        """Load watershed map data from MATLAB outputs and apply IDENTICAL preprocessing"""
        print("\n4. Loading watershed map data and applying MATLAB preprocessing...")
        
        # Load watershed map file
        watershed_file = 'watershed_SNI_TBI.mat'
        if not Path(watershed_file).exists():
            print(f"⚠️  Watershed file not found: {watershed_file}")
            print("   This script will create bar plots instead of spatial heatmaps")
            self.watershed_data = None
            return None
        
        try:
            mat_data = loadmat(watershed_file)
            D_orig = mat_data['D']
            LL2_orig = mat_data['LL2'] 
            llbwb_orig = mat_data['llbwb']
            
            print(f"✅ Loaded original watershed data: density map {D_orig.shape}")
            print(f"   Original watershed regions: {len(np.unique(LL2_orig[LL2_orig > 0]))}")
            
            # Apply IDENTICAL MATLAB preprocessing parameters
            print("🔄 Applying MATLAB preprocessing (opts.resegmentEnable = true)...")
            
            # MATLAB parameters (IDENTICAL to analyze script)
            opts = {
                'resegmentGamma': 1.7,          # opts.resegmentGamma = 1.7
                'resegmentMinDensity': 5e-6,    # opts.resegmentMinDensity = 5e-6
                'resegmentConnectivity': 4,     # opts.resegmentConnectivity = 4
                'resegmentFillHoles': True,     # opts.resegmentFillHoles = true
                'resegmentMinRegionSize': 10,   # opts.resegmentMinRegionSize = 10
                'forceAllBoundaries': True      # opts.forceAllBoundaries = true
            }
            
            # Step 1: Apply gamma correction - MATLAB: Dw = Dw .^ opts.resegmentGamma
            print(f"   Applying gamma correction: {opts['resegmentGamma']}")
            Dw = D_orig ** opts['resegmentGamma']
            print(f"   After gamma D range: [{np.min(Dw):.6f}, {np.max(Dw):.6f}]")
            
            # Step 2: Watershed re-segmentation - MATLAB: LL = watershed(-Dw, opts.resegmentConnectivity)
            print(f"   Running NEW watershed segmentation with connectivity={opts['resegmentConnectivity']}")
            from skimage.segmentation import watershed
            from scipy import ndimage
            
            # MATLAB runs watershed on NEGATIVE of gamma-corrected density
            # This creates COMPLETELY NEW regions, not using original LL2!
            LL = watershed(-Dw)  # watershed on negative creates regions at local maxima
            LL2 = LL.copy()
            print(f"   Created NEW watershed regions: {len(np.unique(LL[LL > 0]))} regions")
            
            # Step 3: Mask low densities as background - MATLAB: LL2(backgroundMask) = -1
            background_mask = Dw < opts['resegmentMinDensity']
            LL2[background_mask] = -1
            print(f"   Background pixels (D<{opts['resegmentMinDensity']:.2e}): {np.sum(background_mask)} ({100*np.sum(background_mask)/Dw.size:.1f}%)")
            
            # Step 4: Fill holes and remove small regions (simplified)
            if opts['resegmentFillHoles'] or opts['resegmentMinRegionSize'] > 0:
                print("   Filling holes and removing small regions...")
                unique_regions = np.unique(LL2[LL2 > 0])
                for region_id in unique_regions:
                    region_mask = (LL2 == region_id)
                    if np.sum(region_mask) < opts['resegmentMinRegionSize']:
                        LL2[region_mask] = -1  # Remove small regions
            
            # Step 5: Renumber regions consecutively 1-24 to match frequency data
            print("   Renumbering regions consecutively...")
            unique_regions = np.unique(LL2[LL2 > 0])
            print(f"   Original region IDs: {unique_regions[:10]}..." if len(unique_regions) > 10 else f"   Original region IDs: {unique_regions}")
            
            # Create mapping from old IDs to consecutive 1-24
            LL2_consecutive = np.zeros_like(LL2)
            LL2_consecutive[LL2 <= 0] = -1  # Keep background as -1
            
            for new_id, old_id in enumerate(unique_regions, start=1):
                LL2_consecutive[LL2 == old_id] = new_id
            
            LL2 = LL2_consecutive
            print(f"   Renumbered to consecutive 1-{len(unique_regions)} regions")
            
            # Step 6: Recompute boundaries - MATLAB boundary computation
            print("   Recomputing boundaries...")
            from scipy import ndimage
            # Create boundary map
            boundary_map = np.zeros_like(LL2, dtype=bool)
            
            for region_id in range(1, len(unique_regions) + 1):
                region_mask = (LL2 == region_id)
                # Get boundary using erosion (MATLAB equivalent)
                eroded = ndimage.binary_erosion(region_mask)
                boundary = region_mask & ~eroded
                boundary_map |= boundary
            
            # Extract boundary coordinates (row, col format like MATLAB)
            boundary_coords = np.column_stack(np.where(boundary_map))
            
            # Store processed data
            self.watershed_data = {
                'D': Dw,                    # PROCESSED density map (with gamma)
                'LL2': LL2,                 # PROCESSED watershed labels
                'llbwb': boundary_coords,   # PROCESSED boundary points  
                'xx': mat_data['xx'],       # Original coordinate grid
                'D_orig': D_orig,           # Keep original for reference
                'LL2_orig': LL2_orig,       # Keep original for reference
                'preprocessing_opts': opts  # Store parameters
            }
            
            # DEBUG: Final region check
            final_unique_regions = np.unique(LL2[LL2 > 0])
            print(f"✅ Applied MATLAB preprocessing:")
            print(f"   Processed regions: {len(final_unique_regions)}")
            print(f"   🔍 DEBUG - Final region IDs in LL2: {sorted(final_unique_regions)}")
            print(f"   Processed boundary points: {boundary_coords.shape}")
            print(f"   Gamma-corrected density range: [{np.min(Dw):.6f}, {np.max(Dw):.6f}]")
            
            # Double check region counts
            for r in range(1, 25):
                count = np.sum(LL2 == r)
                if count > 0:
                    print(f"      Region {r}: {count} pixels")
                else:
                    print(f"      Region {r}: MISSING!")
            
            return self.watershed_data
            
        except Exception as e:
            print(f"❌ Error loading and processing watershed data: {e}")
            import traceback
            traceback.print_exc()
            self.watershed_data = None
            return None
    
    def create_region_frequency_bar_plot(self):
        """Create bar plot comparing region frequencies between groups"""
        print("\n5. Creating region frequency bar plot...")
        
        if not hasattr(self, 'group_stats') or not hasattr(self, 'difference_stats'):
            print("❌ Group statistics not calculated")
            return
        
        # Prepare data for plotting
        regions = sorted(list(self.difference_stats.keys()))
        other_means = [self.difference_stats[r]['other_mean'] for r in regions]
        tbi_means = [self.difference_stats[r]['tbi_mean'] for r in regions]
        differences = [self.difference_stats[r]['difference'] for r in regions]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Group Frequency Analysis: Other Groups vs TBI Groups', fontsize=16, fontweight='bold')
        
        # 1. Bar plot comparing group means
        ax1 = axes[0, 0]
        x = np.arange(len(regions))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, other_means, width, label='Other Groups', alpha=0.8, color='steelblue')
        bars2 = ax1.bar(x + width/2, tbi_means, width, label='TBI Groups', alpha=0.8, color='orange')
        
        ax1.set_xlabel('Watershed Region')
        ax1.set_ylabel('Mean Frequency (frames/total)')
        ax1.set_title('Mean Frequency by Group')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'R{r}' for r in regions], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Difference plot (Other - TBI)
        ax2 = axes[0, 1]
        colors = ['red' if d < 0 else 'blue' for d in differences]
        bars = ax2.bar(x, differences, alpha=0.8, color=colors)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Watershed Region')
        ax2.set_ylabel('Frequency Difference (Other - TBI)')
        ax2.set_title('Frequency Differences')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'R{r}' for r in regions], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add legend for difference colors
        red_patch = mpatches.Patch(color='red', label='Higher in TBI')
        blue_patch = mpatches.Patch(color='blue', label='Higher in Other')
        ax2.legend(handles=[blue_patch, red_patch])
        
        # 3. Ranked difference plot
        ax3 = axes[1, 0]
        sorted_indices = np.argsort(differences)
        sorted_regions = [regions[i] for i in sorted_indices]
        sorted_differences = [differences[i] for i in sorted_indices]
        sorted_colors = ['red' if d < 0 else 'blue' for d in sorted_differences]
        
        y_pos = np.arange(len(sorted_regions))
        bars = ax3.barh(y_pos, sorted_differences, alpha=0.8, color=sorted_colors)
        
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f'R{r}' for r in sorted_regions])
        ax3.set_xlabel('Frequency Difference (Other - TBI)')
        ax3.set_title('Ranked Frequency Differences')
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate summary statistics
        n_other = len(self.group_frequencies[self.group_frequencies['group_type'] == 'Other'])
        n_tbi = len(self.group_frequencies[self.group_frequencies['group_type'] == 'TBI'])
        n_regions = len(regions)
        mean_abs_diff = np.mean(np.abs(differences))
        max_pos_diff = max([d for d in differences if d > 0]) if any(d > 0 for d in differences) else 0
        max_neg_diff = min([d for d in differences if d < 0]) if any(d < 0 for d in differences) else 0
        
        summary_text = f"""
SUMMARY STATISTICS

Groups:
• Other Groups: {n_other} videos
• TBI Groups: {n_tbi} videos
• Total Regions: {n_regions}

Differences (Other - TBI):
• Mean Absolute Difference: {mean_abs_diff:.4f}
• Maximum Positive (Other > TBI): {max_pos_diff:.4f}
• Maximum Negative (TBI > Other): {max_neg_diff:.4f}

Regions with Highest Other Activity:
• {sorted_regions[-1]} (R{sorted_regions[-1]}): +{sorted_differences[-1]:.4f}
• {sorted_regions[-2]} (R{sorted_regions[-2]}): +{sorted_differences[-2]:.4f}

Regions with Highest TBI Activity:
• {sorted_regions[0]} (R{sorted_regions[0]}): {sorted_differences[0]:.4f}
• {sorted_regions[1]} (R{sorted_regions[1]}): {sorted_differences[1]:.4f}
        """.strip()
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.heatmap_output_dir / 'group_frequency_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved bar plot: {output_path}")
        
        plt.show()
        
        return fig
    
    def create_spatial_difference_heatmap(self):
        """Create spatial heatmap showing difference patterns on the behavioral map"""
        print("\n6. Creating spatial difference heatmap...")
        
        if self.watershed_data is None:
            print("❌ Watershed data not available - cannot create spatial heatmap")
            return None
        
        if not hasattr(self, 'difference_stats'):
            print("❌ Difference statistics not calculated")
            return None
        
        # Create difference map by assigning difference values to watershed regions
        D = self.watershed_data['D']
        LL2 = self.watershed_data['LL2']
        
        # DEBUG: Check what regions we have
        print("\n🔍 DEBUG: Spatial Heatmap Creation")
        unique_regions_in_LL2 = np.unique(LL2[LL2 > 0])
        print(f"   Regions in LL2 watershed map: {sorted(unique_regions_in_LL2)}")
        print(f"   Number of regions in LL2: {len(unique_regions_in_LL2)}")
        
        print(f"   Regions in difference_stats: {sorted(self.difference_stats.keys())}")
        print(f"   Number of regions in difference_stats: {len(self.difference_stats)}")
        
        # Initialize difference map
        difference_map = np.zeros_like(D, dtype=np.float64)
        
        # Assign difference values to each watershed region
        regions_mapped = 0
        regions_not_found = []
        for region, stats in self.difference_stats.items():
            mask = (LL2 == region)
            if np.any(mask):
                difference_map[mask] = stats['difference']
                regions_mapped += 1
            else:
                regions_not_found.append(region)
        
        print(f"   Regions successfully mapped: {regions_mapped}")
        if regions_not_found:
            print(f"   ⚠️  Regions NOT found in LL2: {regions_not_found}")
        
        # Create custom colormap: blue for Other > TBI, red for TBI > Other
        colors = ['darkred', 'red', 'white', 'blue', 'darkblue']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('difference', colors, N=n_bins)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Spatial Frequency Differences: Other Groups vs TBI Groups', 
                     fontsize=16, fontweight='bold')
        
        # 1. Original density map - IDENTICAL to MATLAB
        ax1 = axes[0, 0]
        # MATLAB: imagesc(D); axis equal off; colormap(flipud(gray)); caxis([0 max(D(:))*0.8]);
        vmax_density = np.max(D) * 0.8  # caxis([0 max(D(:))*0.8])
        im1 = ax1.imshow(D, cmap='gray_r', aspect='equal', vmin=0, vmax=vmax_density, 
                        origin='upper', interpolation='nearest')  # imagesc equivalent
        
        # MATLAB: scatter(llbwb(:,2), llbwb(:,1), 1, 'k', '.');
        if 'llbwb' in self.watershed_data:
            llbwb = self.watershed_data['llbwb']
            # MATLAB indexing: llbwb(:,2)=x, llbwb(:,1)=y, size=1, color='k', marker='.'
            ax1.scatter(llbwb[:, 1], llbwb[:, 0], s=1, c='black', marker='.', linewidths=0)
        
        ax1.set_title('Behavioral Density Map (IDENTICAL to MATLAB)')
        ax1.axis('off')  # axis equal off
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Density')
        
        # 2. Watershed regions with labels
        ax2 = axes[0, 1]
        # Create colored watershed map
        watershed_colored = np.zeros((*LL2.shape, 3))
        unique_regions = np.unique(LL2[LL2 > 0])
        
        # Use distinct colors for 24 regions
        if len(unique_regions) <= 20:
            colors_regions = plt.cm.tab20(np.linspace(0, 1, len(unique_regions)))
        else:
            # Use a combination of colormaps for more than 20 regions
            colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
            colors2 = plt.cm.Set3(np.linspace(0, 1, len(unique_regions) - 20))
            colors_regions = np.vstack([colors1, colors2])
        
        print(f"\n🔍 DEBUG: Visualizing {len(unique_regions)} watershed regions")
        
        for i, region in enumerate(unique_regions):
            mask = (LL2 == region)
            watershed_colored[mask] = colors_regions[i % len(colors_regions), :3]
            
            # Find centroid for label
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
                # Add text label to identify region
                ax2.text(cx, cy, str(region), color='black', fontsize=8, 
                        ha='center', va='center', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax2.imshow(watershed_colored)
        ax2.set_title(f'Watershed Regions (Total: {len(unique_regions)})')
        ax2.axis('off')
        
        # 3. Difference heatmap
        ax3 = axes[1, 0]
        
        # Mask background (LL2 <= 0) as white
        masked_diff = np.ma.masked_where(LL2 <= 0, difference_map)
        
        # Create normalization that centers at 0
        vmax = max(abs(np.min(difference_map[LL2 > 0])), abs(np.max(difference_map[LL2 > 0])))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        im3 = ax3.imshow(masked_diff, cmap=cmap, norm=norm, aspect='equal')
        ax3.set_title('Frequency Differences\n(Blue: Other > TBI, Red: TBI > Other)')
        ax3.axis('off')
        
        # Add colorbar
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, label='Frequency Difference')
        cbar3.set_label('Other - TBI Frequency', rotation=270, labelpad=20)
        
        # 4. Overlay on density map - IDENTICAL to MATLAB base
        ax4 = axes[1, 1]
        
        # Show density map as background - IDENTICAL to MATLAB
        vmax_density = np.max(D) * 0.8
        ax4.imshow(D, cmap='gray_r', alpha=0.4, aspect='equal', vmin=0, vmax=vmax_density,
                  origin='upper', interpolation='nearest')
        
        # Add watershed boundaries - IDENTICAL to MATLAB
        if 'llbwb' in self.watershed_data:
            llbwb = self.watershed_data['llbwb']
            ax4.scatter(llbwb[:, 1], llbwb[:, 0], s=1, c='black', marker='.', 
                       alpha=0.6, linewidths=0)
        
        # Overlay difference map with transparency
        im4 = ax4.imshow(masked_diff, cmap=cmap, norm=norm, alpha=0.9, aspect='equal',
                        origin='upper', interpolation='nearest')
        ax4.set_title('Differences Overlaid on MATLAB-Identical Behavioral Map')
        ax4.axis('off')
        
        # Add colorbar
        cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8, label='Frequency Difference')
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.heatmap_output_dir / 'spatial_difference_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved spatial heatmap: {output_path}")
        
        plt.show()
        
        return fig, difference_map
    
    def export_difference_data(self):
        """Export difference statistics to CSV files"""
        print("\n7. Exporting difference data...")
        
        if not hasattr(self, 'difference_stats'):
            print("❌ No difference statistics to export")
            return
        
        # Export difference statistics
        diff_data = []
        for region, stats in self.difference_stats.items():
            diff_data.append({
                'Region': region,
                'Other_Mean_Frequency': stats['other_mean'],
                'TBI_Mean_Frequency': stats['tbi_mean'],
                'Difference_Other_minus_TBI': stats['difference'],
                'Other_Group_N_Videos': stats['other_n'],
                'TBI_Group_N_Videos': stats['tbi_n'],
                'Abs_Difference': abs(stats['difference']),
                'Favors_Group': 'Other' if stats['difference'] > 0 else 'TBI'
            })
        
        diff_df = pd.DataFrame(diff_data)
        diff_df = diff_df.sort_values('Difference_Other_minus_TBI', ascending=False)
        
        # Save difference statistics
        diff_file = self.heatmap_output_dir / 'region_frequency_differences.csv'
        diff_df.to_csv(diff_file, index=False)
        print(f"✅ Saved difference statistics: {diff_file}")
        
        # Export detailed group statistics
        group_data = []
        for group_type in ['Other', 'TBI']:
            if group_type in self.group_stats:
                for region, stats in self.group_stats[group_type].items():
                    group_data.append({
                        'Group_Type': group_type,
                        'Region': region,
                        'Mean_Frequency': stats['mean'],
                        'Std_Frequency': stats['std'],
                        'SEM_Frequency': stats['sem'],
                        'Median_Frequency': stats['median'],
                        'N_Videos': stats['n_videos']
                    })
        
        group_df = pd.DataFrame(group_data)
        group_file = self.heatmap_output_dir / 'detailed_group_statistics.csv'
        group_df.to_csv(group_file, index=False)
        print(f"✅ Saved detailed group statistics: {group_file}")
        
        # Export individual video classifications
        video_file = self.heatmap_output_dir / 'video_group_classifications.csv'
        self.group_frequencies.to_csv(video_file, index=False)
        print(f"✅ Saved video classifications: {video_file}")
        
        return diff_df, group_df
    
    def create_summary_report(self):
        """Create a text summary report"""
        print("\n8. Creating summary report...")
        
        report_file = self.heatmap_output_dir / 'group_difference_analysis_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("GROUP FREQUENCY DIFFERENCE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Directory: {self.analysis_outputs_dir}\n\n")
            
            # Group information
            f.write("GROUP DEFINITIONS:\n")
            f.write("-" * 20 + "\n")
            other_groups = self.group_frequencies[self.group_frequencies['group_type'] == 'Other']['group'].unique()
            tbi_groups = self.group_frequencies[self.group_frequencies['group_type'] == 'TBI']['group'].unique()
            
            f.write(f"Other Groups: {', '.join(sorted(other_groups))}\n")
            f.write(f"TBI Groups: {', '.join(sorted(tbi_groups))}\n\n")
            
            n_other = len(self.group_frequencies[self.group_frequencies['group_type'] == 'Other'])
            n_tbi = len(self.group_frequencies[self.group_frequencies['group_type'] == 'TBI'])
            
            f.write(f"Total Videos - Other Groups: {n_other}\n")
            f.write(f"Total Videos - TBI Groups: {n_tbi}\n")
            f.write(f"Total Regions Analyzed: {len(self.difference_stats)}\n\n")
            
            # Top differences
            f.write("TOP FREQUENCY DIFFERENCES (Other - TBI):\n")
            f.write("-" * 40 + "\n")
            
            # Sort regions by difference magnitude
            sorted_diffs = sorted(self.difference_stats.items(), 
                                key=lambda x: x[1]['difference'], reverse=True)
            
            f.write("Regions with HIGHER frequency in Other groups:\n")
            for region, stats in sorted_diffs[:5]:
                if stats['difference'] > 0:
                    f.write(f"  Region {region}: +{stats['difference']:.4f} "
                           f"(Other: {stats['other_mean']:.4f}, TBI: {stats['tbi_mean']:.4f})\n")
            
            f.write("\nRegions with HIGHER frequency in TBI groups:\n")
            for region, stats in reversed(sorted_diffs[-5:]):
                if stats['difference'] < 0:
                    f.write(f"  Region {region}: {stats['difference']:.4f} "
                           f"(Other: {stats['other_mean']:.4f}, TBI: {stats['tbi_mean']:.4f})\n")
            
            # Summary statistics
            differences = [stats['difference'] for stats in self.difference_stats.values()]
            f.write(f"\nSUMMARY STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mean absolute difference: {np.mean(np.abs(differences)):.4f}\n")
            f.write(f"Maximum positive difference: {max(differences):.4f}\n")
            f.write(f"Maximum negative difference: {min(differences):.4f}\n")
            f.write(f"Standard deviation of differences: {np.std(differences):.4f}\n")
        
        print(f"✅ Saved summary report: {report_file}")
        return report_file
    
    def run_complete_analysis(self):
        """Run the complete group difference analysis"""
        try:
            print("Starting group difference heatmap analysis...")
            
            # Load and process data
            self.load_frequency_data()
            self.group_videos_by_type()
            self.load_watershed_map_data()
            
            # Create visualizations
            self.create_region_frequency_bar_plot()
            
            # Create spatial heatmap if watershed data available
            if self.watershed_data is not None:
                self.create_spatial_difference_heatmap()
            
            # Export results
            self.export_difference_data()
            self.create_summary_report()
            
            print("\n" + "="*60)
            print("GROUP DIFFERENCE ANALYSIS COMPLETE")
            print("="*60)
            
            # Print key findings
            self.print_key_findings()
            
        except Exception as e:
            print(f"❌ Error in analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def print_key_findings(self):
        """Print key findings to console"""
        print("\nKEY FINDINGS:")
        print("-" * 15)
        
        if not hasattr(self, 'difference_stats'):
            print("No difference statistics available")
            return
        
        # Sort regions by difference (Other - TBI)
        sorted_diffs = sorted(self.difference_stats.items(), 
                            key=lambda x: x[1]['difference'], reverse=True)
        
        print("\n🔵 Regions with HIGHER activity in Other groups:")
        positive_count = 0
        for region, stats in sorted_diffs:
            if stats['difference'] > 0.001:  # Threshold for meaningful difference
                print(f"  Region {region}: +{stats['difference']:.4f} "
                     f"(Other: {stats['other_mean']:.4f}, TBI: {stats['tbi_mean']:.4f})")
                positive_count += 1
            if positive_count >= 5:  # Show top 5
                break
        
        print("\n🔴 Regions with HIGHER activity in TBI groups:")
        negative_count = 0
        for region, stats in reversed(sorted_diffs):
            if stats['difference'] < -0.001:  # Threshold for meaningful difference
                print(f"  Region {region}: {stats['difference']:.4f} "
                     f"(Other: {stats['other_mean']:.4f}, TBI: {stats['tbi_mean']:.4f})")
                negative_count += 1
            if negative_count >= 5:  # Show top 5
                break
        
        # Overall statistics
        differences = [stats['difference'] for stats in self.difference_stats.values()]
        n_other_higher = sum(1 for d in differences if d > 0.001)
        n_tbi_higher = sum(1 for d in differences if d < -0.001)
        
        print(f"\n📊 Summary:")
        print(f"  Regions with higher Other activity: {n_other_higher}")
        print(f"  Regions with higher TBI activity: {n_tbi_higher}")
        print(f"  Mean absolute difference: {np.mean(np.abs(differences)):.4f}")


def main():
    """Main execution function"""
    print("Starting group difference heatmap analysis...")
    
    # Initialize analyzer
    analyzer = GroupDifferenceHeatmapGenerator('analysis_outputs')
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    print(f"\nResults saved to: {analyzer.heatmap_output_dir}")
    print("Files created:")
    print("  - group_frequency_comparison.png (bar plots)")
    print("  - spatial_difference_heatmap.png (spatial heatmaps, if watershed data available)")
    print("  - region_frequency_differences.csv (difference statistics)")
    print("  - detailed_group_statistics.csv (group statistics)")
    print("  - video_group_classifications.csv (individual video data)")
    print("  - group_difference_analysis_report.txt (summary report)")


if __name__ == "__main__":
    main()
