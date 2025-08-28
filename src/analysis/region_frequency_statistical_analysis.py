#!/usr/bin/env python3
"""
Statistical Analysis of Region Frequencies

This script:
1. Loads region count data from MATLAB analysis outputs
2. Sums frequencies for specific regions (2, 3, 4, 6, 7, 8) per video
3. Groups videos by experimental condition (treating week4 as separate groups)
4. Performs pairwise statistical comparisons between groups
5. Creates visualizations and exports results

Author: Generated for behavioral embedding analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, ttest_ind
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RegionFrequencyAnalysis:
    """Statistical analysis of region frequencies across experimental groups"""
    
    def __init__(self, analysis_outputs_dir='analysis_outputs'):
        """
        Initialize the analysis
        
        Args:
            analysis_outputs_dir: Directory containing MATLAB analysis outputs
        """
        self.analysis_outputs_dir = Path(analysis_outputs_dir)
        self.csv_dir = self.analysis_outputs_dir / 'csv'
        self.output_dir = self.analysis_outputs_dir / 'statistical_analysis'
        self.output_dir.mkdir(exist_ok=True)
        
        # Regions of interest (1-indexed as in MATLAB output)
        #self.regions_of_interest = [2, 3, 5, 6, 8,9]
        self.regions_of_interest = [2, 3, 5, 6]
        # Data storage
        self.region_counts = None
        self.total_frames_per_video = None
        self.video_frequencies = None
        self.video_metadata = None
        self.group_data = None
        self.statistical_results = {}
        
        print("="*60)
        print("REGION FREQUENCY STATISTICAL ANALYSIS")
        print("="*60)
        print(f"Analysis outputs directory: {self.analysis_outputs_dir}")
        print(f"Regions of interest: {self.regions_of_interest}")
        print(f"Output directory: {self.output_dir}")
    
    def parse_filename_to_group(self, filename):
        """
        Parse filename to extract experimental group
        
        Args:
            filename: e.g., 'DRG_1.mat', 'week4-DRG_2.mat'
            
        Returns:
            group: experimental group with week designation
        """
        # Remove .mat extension if present
        name = filename.replace('.mat', '')
        print(f"Debug: parsing '{filename}' -> base name: '{name}'")
        
        # Check if it's a week4 file
        if name.startswith('week4-'):
            week = 'week4'
            name = name[6:]  # Remove 'week4-' prefix (6 characters, not 7!)
            print(f"Debug: week4 file, remaining name: '{name}'")
        else:
            week = 'week1'
            print(f"Debug: week1 file, name: '{name}'")
        
        # Extract base group (handle underscore separation)
        for group in ['DRG', 'IT', 'SC', 'SNI', 'TBI']:
            if name.startswith(group):
                if week == 'week4':
                    result = f'week4-{group}'
                else:
                    result = group
                print(f"Debug: matched group '{group}' -> final group: '{result}'")
                return result
        
        print(f"Debug: no group match found for '{name}' - returning 'unknown'")
        return 'unknown'
    
    def load_region_counts(self):
        """Load region count data from CSV file"""
        print("\n1. Loading region count data...")
        
        counts_file = self.csv_dir / 'per_file_region_counts.csv'
        if not counts_file.exists():
            raise FileNotFoundError(f"Region counts file not found: {counts_file}")
        
        # Load the data
        self.region_counts = pd.read_csv(counts_file)
        print(f"✅ Loaded region counts: {self.region_counts.shape}")
        print(f"   Files: {len(self.region_counts)}")
        print(f"   Region columns: {len([col for col in self.region_counts.columns if col.startswith('Region_')])}")
        
        # Check which region columns are available
        available_regions = [int(col.split('_')[1]) for col in self.region_counts.columns if col.startswith('Region_')]
        missing_regions = [r for r in self.regions_of_interest if r not in available_regions]
        
        print(f"   Available regions: {sorted(available_regions)}")
        if missing_regions:
            print(f"   ⚠️  Missing regions: {missing_regions}")
            # Filter to only available regions
            self.regions_of_interest = [r for r in self.regions_of_interest if r in available_regions]
            print(f"   Using available regions: {self.regions_of_interest}")
        
        return self.region_counts
    
    def calculate_total_frames_per_video(self):
        """Calculate total frames for each video from frame indices files"""
        print("\n2. Calculating total frames per video...")
        
        frame_indices_dir = self.csv_dir / 'frame_indices_per_video'
        if not frame_indices_dir.exists():
            raise FileNotFoundError(f"Frame indices directory not found: {frame_indices_dir}")
        
        total_frames = {}
        
        for idx, row in self.region_counts.iterrows():
            filename = row['File']
            # Find corresponding frame indices file
            pattern = f"*_{filename}_frame_indices.csv"
            matching_files = list(frame_indices_dir.glob(pattern))
            
            if not matching_files:
                print(f"   ⚠️  No frame indices file found for {filename}")
                # Try alternative pattern
                pattern2 = f"*{filename.replace('.mat', '')}.mat_frame_indices.csv"
                matching_files = list(frame_indices_dir.glob(pattern2))
                
            if matching_files:
                indices_file = matching_files[0]
                try:
                    # Read frame indices and find maximum frame number
                    frame_data = pd.read_csv(indices_file)
                    # Get maximum frame number across all regions (excluding NaN)
                    max_frame = 0
                    for col in frame_data.columns:
                        col_max = frame_data[col].replace([np.inf, -np.inf], np.nan).dropna().max()
                        if not pd.isna(col_max) and col_max > max_frame:
                            max_frame = col_max
                    
                    total_frames[filename] = int(max_frame) if max_frame > 0 else 1
                    
                except Exception as e:
                    print(f"   ⚠️  Error reading {indices_file}: {e}")
                    total_frames[filename] = 1  # Default fallback
            else:
                print(f"   ⚠️  No frame indices file found for {filename}, using default")
                total_frames[filename] = 1  # Default fallback
        
        self.total_frames_per_video = total_frames
        print(f"✅ Calculated total frames for {len(total_frames)} videos")
        
        # Show some statistics
        frame_counts = list(total_frames.values())
        if frame_counts:
            print(f"   Frame count range: {min(frame_counts)} - {max(frame_counts)}")
            print(f"   Mean frames: {np.mean(frame_counts):.0f} ± {np.std(frame_counts):.0f}")
        
        return self.total_frames_per_video
    
    def calculate_video_frequencies(self):
        """Calculate frequency ratios (region_count / total_frames) for each video and region"""
        print("\n3. Calculating frequency ratios...")
        
        frequency_data = []
        
        for idx, row in self.region_counts.iterrows():
            filename = row['File']
            total_frames = self.total_frames_per_video.get(filename, 1)
            
            freq_row = {'filename': filename, 'total_frames': total_frames}
            
            # Calculate frequency for each region
            for region in range(1, 25):  # Assuming regions 1-24
                region_col = f'Region_{region}'
                if region_col in self.region_counts.columns:
                    count = row[region_col]
                    frequency = count / total_frames if total_frames > 0 else 0
                    freq_row[f'Freq_Region_{region}'] = frequency
                    freq_row[region_col] = count  # Keep original count for reference
            
            frequency_data.append(freq_row)
        
        self.video_frequencies = pd.DataFrame(frequency_data)
        print(f"✅ Calculated frequencies for {len(self.video_frequencies)} videos")
        
        return self.video_frequencies
    
    def create_video_metadata(self):
        """Create metadata for each video including group assignment"""
        print("\n4. Creating video metadata...")
        
        metadata_list = []
        
        for idx, row in self.video_frequencies.iterrows():
            filename = row['filename']
            group = self.parse_filename_to_group(filename)
            total_frames = row['total_frames']
            
            # Sum frequencies for regions of interest (using frequency ratios now)
            freq_columns = [f'Freq_Region_{r}' for r in self.regions_of_interest]
            available_freq_columns = [col for col in freq_columns if col in self.video_frequencies.columns]
            
            if available_freq_columns:
                total_frequency = row[available_freq_columns].sum()
            else:
                total_frequency = 0
            
            metadata_list.append({
                'filename': filename,
                'group': group,
                'total_frequency': total_frequency,  # Now this is sum of frequency ratios
                'total_frames': total_frames,
                'video_index': idx
            })
        
        self.video_metadata = pd.DataFrame(metadata_list)
        
        # Display group distribution
        group_counts = self.video_metadata['group'].value_counts()
        print(f"✅ Created metadata for {len(self.video_metadata)} videos")
        print("\n   Group distribution:")
        for group, count in group_counts.items():
            group_freq = self.video_metadata[self.video_metadata['group'] == group]['total_frequency']
            print(f"     {group}: {count} videos (mean freq ratio: {group_freq.mean():.4f} ± {group_freq.std():.4f})")
        
        return self.video_metadata
    
    def prepare_group_data(self):
        """Prepare data grouped by experimental condition"""
        print("\n5. Preparing group data...")
        
        # Group data by experimental condition
        self.group_data = {}
        
        for group in self.video_metadata['group'].unique():
            if group == 'unknown':
                continue
                
            group_videos = self.video_metadata[self.video_metadata['group'] == group]
            frequencies = group_videos['total_frequency'].values
            
            self.group_data[group] = {
                'frequencies': frequencies,
                'n_videos': len(frequencies),
                'mean': np.mean(frequencies),
                'std': np.std(frequencies),
                'median': np.median(frequencies),
                'sem': np.std(frequencies) / np.sqrt(len(frequencies))
            }
            
            print(f"   {group}: {len(frequencies)} videos, mean={np.mean(frequencies):.4f} ± {np.std(frequencies):.4f}")
        
        return self.group_data
    
    def perform_pairwise_tests(self, test_type='mannwhitney'):
        """
        Perform pairwise statistical tests between all groups
        
        Args:
            test_type: 'mannwhitney', 'ttest', or 'both'
        """
        print(f"\n6. Performing pairwise statistical tests ({test_type})...")
        
        groups = list(self.group_data.keys())
        n_groups = len(groups)
        
        if n_groups < 2:
            print("❌ Need at least 2 groups for pairwise testing")
            return
        
        # Initialize results storage
        self.statistical_results = {
            'pairwise_tests': [],
            'summary': {
                'n_groups': n_groups,
                'groups': groups,
                'test_type': test_type
            }
        }
        
        print(f"   Testing {n_groups} groups: {groups}")
        print(f"   Total comparisons: {n_groups * (n_groups - 1) // 2}")
        
        # Perform all pairwise comparisons
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups[i+1:], i+1):
                
                data1 = self.group_data[group1]['frequencies']
                data2 = self.group_data[group2]['frequencies']
                
                # Prepare result dict
                result = {
                    'group1': group1,
                    'group2': group2,
                    'n1': len(data1),
                    'n2': len(data2),
                    'mean1': np.mean(data1),
                    'mean2': np.mean(data2),
                    'std1': np.std(data1),
                    'std2': np.std(data2)
                }
                
                # Mann-Whitney U test (non-parametric)
                if test_type in ['mannwhitney', 'both']:
                    try:
                        statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                        result['mannwhitney_statistic'] = statistic
                        result['mannwhitney_p'] = p_value
                        result['mannwhitney_significant'] = p_value < 0.05
                    except Exception as e:
                        print(f"   Warning: Mann-Whitney test failed for {group1} vs {group2}: {e}")
                        result['mannwhitney_p'] = np.nan
                        result['mannwhitney_significant'] = False
                
                # T-test (parametric)
                if test_type in ['ttest', 'both']:
                    try:
                        statistic, p_value = ttest_ind(data1, data2, equal_var=False)  # Welch's t-test
                        result['ttest_statistic'] = statistic
                        result['ttest_p'] = p_value
                        result['ttest_significant'] = p_value < 0.05
                    except Exception as e:
                        print(f"   Warning: T-test failed for {group1} vs {group2}: {e}")
                        result['ttest_p'] = np.nan
                        result['ttest_significant'] = False
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(data1)-1)*np.std(data1)**2 + (len(data2)-1)*np.std(data2)**2) / (len(data1)+len(data2)-2))
                if pooled_std > 0:
                    result['cohens_d'] = (np.mean(data1) - np.mean(data2)) / pooled_std
                else:
                    result['cohens_d'] = 0
                
                self.statistical_results['pairwise_tests'].append(result)
        
        # Overall test (Kruskal-Wallis for multiple groups)
        if n_groups > 2:
            try:
                all_data = [self.group_data[group]['frequencies'] for group in groups]
                kw_statistic, kw_p = kruskal(*all_data)
                self.statistical_results['overall_test'] = {
                    'test': 'Kruskal-Wallis',
                    'statistic': kw_statistic,
                    'p_value': kw_p,
                    'significant': kw_p < 0.05
                }
                print(f"   Overall Kruskal-Wallis test: p = {kw_p:.6f}")
            except Exception as e:
                print(f"   Warning: Kruskal-Wallis test failed: {e}")
        
        # Summary statistics
        if test_type in ['mannwhitney', 'both']:
            significant_mw = sum(1 for r in self.statistical_results['pairwise_tests'] if r.get('mannwhitney_significant', False))
            print(f"   Mann-Whitney significant pairs: {significant_mw}/{len(self.statistical_results['pairwise_tests'])}")
        
        if test_type in ['ttest', 'both']:
            significant_tt = sum(1 for r in self.statistical_results['pairwise_tests'] if r.get('ttest_significant', False))
            print(f"   T-test significant pairs: {significant_tt}/{len(self.statistical_results['pairwise_tests'])}")
        
        return self.statistical_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n7. Creating visualizations...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Region Frequency Analysis\nRegions: {self.regions_of_interest}', fontsize=16)
        
        # 1. Box plot by group
        ax1 = axes[0, 0]
        groups = list(self.group_data.keys())
        frequencies = [self.group_data[group]['frequencies'] for group in groups]
        
        bp = ax1.boxplot(frequencies, labels=groups, patch_artist=True)
        colors = sns.color_palette("husl", len(groups))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Total Frequency Ratio (Regions of Interest)')
        ax1.set_title('Frequency Distribution by Group')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Violin plot
        ax2 = axes[0, 1]
        
        # Prepare data for seaborn
        plot_data = []
        for group in groups:
            for freq in self.group_data[group]['frequencies']:
                plot_data.append({'Group': group, 'Frequency': freq})
        plot_df = pd.DataFrame(plot_data)
        
        sns.violinplot(data=plot_df, x='Group', y='Frequency', ax=ax2)
        ax2.set_title('Frequency Density by Group')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Statistical significance heatmap
        ax3 = axes[1, 0]
        
        # Create p-value matrix
        n_groups = len(groups)
        p_matrix = np.ones((n_groups, n_groups))
        
        for result in self.statistical_results['pairwise_tests']:
            i = groups.index(result['group1'])
            j = groups.index(result['group2'])
            p_val = result.get('mannwhitney_p', 1.0)
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
        
        # Create heatmap
        mask = np.triu(np.ones_like(p_matrix, dtype=bool), k=1)
        im = ax3.imshow(p_matrix, cmap='RdYlBu_r', vmin=0, vmax=0.1)
        ax3.set_xticks(range(n_groups))
        ax3.set_yticks(range(n_groups))
        ax3.set_xticklabels(groups, rotation=45)
        ax3.set_yticklabels(groups)
        ax3.set_title('P-values (Mann-Whitney U)')
        
        # Add text annotations
        for i in range(n_groups):
            for j in range(n_groups):
                if i < j:  # Only upper triangle
                    p_val = p_matrix[i, j]
                    if p_val < 0.001:
                        text = '***'
                    elif p_val < 0.01:
                        text = '**'
                    elif p_val < 0.05:
                        text = '*'
                    else:
                        text = f'{p_val:.3f}'
                    ax3.text(j, i, text, ha='center', va='center', 
                            color='white' if p_val < 0.05 else 'black', fontweight='bold')
        
        plt.colorbar(im, ax=ax3, label='p-value')
        
        # 4. Effect sizes
        ax4 = axes[1, 1]
        
        comparison_names = []
        effect_sizes = []
        p_values = []
        
        for result in self.statistical_results['pairwise_tests']:
            comp_name = f"{result['group1']}\nvs\n{result['group2']}"
            comparison_names.append(comp_name)
            effect_sizes.append(abs(result['cohens_d']))
            p_values.append(result.get('mannwhitney_p', 1.0))
        
        colors = ['red' if p < 0.05 else 'gray' for p in p_values]
        bars = ax4.bar(range(len(effect_sizes)), effect_sizes, color=colors, alpha=0.7)
        
        ax4.set_ylabel("Cohen's d (Effect Size)")
        ax4.set_title('Effect Sizes (Red = Significant)')
        ax4.set_xticks(range(len(comparison_names)))
        ax4.set_xticklabels(comparison_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add horizontal line at effect size thresholds
        ax4.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Small effect')
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Medium effect')
        ax4.axhline(y=0.8, color='darkred', linestyle='--', alpha=0.5, label='Large effect')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / 'region_frequency_analysis.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comprehensive plot: {fig_path}")
        
        plt.show()
    
    def export_results(self):
        """Export statistical results to CSV and text files"""
        print("\n8. Exporting results...")
        
        # Export detailed pairwise results
        results_df = pd.DataFrame(self.statistical_results['pairwise_tests'])
        results_file = self.output_dir / 'pairwise_statistical_tests.csv'
        results_df.to_csv(results_file, index=False)
        print(f"✅ Exported detailed results: {results_file}")
        
        # Export group summary
        group_summary = []
        for group, data in self.group_data.items():
            group_summary.append({
                'Group': group,
                'N_Videos': data['n_videos'],
                'Mean_Frequency': data['mean'],
                'Std_Frequency': data['std'],
                'Median_Frequency': data['median'],
                'SEM': data['sem']
            })
        
        summary_df = pd.DataFrame(group_summary)
        summary_file = self.output_dir / 'group_summary_statistics.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Exported group summary: {summary_file}")
        
        # Export significant pairs only
        significant_pairs = []
        for result in self.statistical_results['pairwise_tests']:
            if result.get('mannwhitney_significant', False):
                significant_pairs.append({
                    'Group1': result['group1'],
                    'Group2': result['group2'],
                    'P_value': result['mannwhitney_p'],
                    'Effect_Size': abs(result['cohens_d']),
                    'Mean1': result['mean1'],
                    'Mean2': result['mean2']
                })
        
        if significant_pairs:
            sig_df = pd.DataFrame(significant_pairs)
            sig_file = self.output_dir / 'significant_pairs.csv'
            sig_df.to_csv(sig_file, index=False)
            print(f"✅ Exported significant pairs: {sig_file}")
        else:
            print("   No significant pairs found")
        
        # Export individual video frequencies
        self._export_video_frequencies()
        
        # Export regions of interest summary per video
        self._export_regions_of_interest_summary()
        
        # Create text summary
        self._create_text_summary()
    
    def _create_text_summary(self):
        """Create a human-readable text summary"""
        summary_file = self.output_dir / 'analysis_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("REGION FREQUENCY STATISTICAL ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Regions Analyzed: {self.regions_of_interest}\n")
            f.write(f"Total Videos: {len(self.video_metadata)}\n\n")
            
            # Group statistics
            f.write("GROUP STATISTICS:\n")
            f.write("-" * 20 + "\n")
            for group, data in self.group_data.items():
                f.write(f"{group}: {data['n_videos']} videos\n")
                f.write(f"  Mean ± SEM: {data['mean']:.2f} ± {data['sem']:.2f}\n")
                f.write(f"  Median: {data['median']:.2f}\n")
                f.write(f"  Range: {np.min(data['frequencies']):.1f} - {np.max(data['frequencies']):.1f}\n\n")
            
            # Significant comparisons
            f.write("SIGNIFICANT PAIRWISE COMPARISONS (p < 0.05):\n")
            f.write("-" * 40 + "\n")
            
            significant_count = 0
            for result in self.statistical_results['pairwise_tests']:
                if result.get('mannwhitney_significant', False):
                    significant_count += 1
                    f.write(f"{result['group1']} vs {result['group2']}:\n")
                    f.write(f"  p = {result['mannwhitney_p']:.6f}\n")
                    f.write(f"  Effect size (Cohen's d) = {abs(result['cohens_d']):.3f}\n")
                    f.write(f"  {result['group1']}: {result['mean1']:.2f} ± {result['std1']:.2f}\n")
                    f.write(f"  {result['group2']}: {result['mean2']:.2f} ± {result['std2']:.2f}\n\n")
            
            if significant_count == 0:
                f.write("No significant differences found.\n\n")
            else:
                f.write(f"Total significant pairs: {significant_count}\n\n")
            
            # Overall test
            if 'overall_test' in self.statistical_results:
                overall = self.statistical_results['overall_test']
                f.write("OVERALL TEST:\n")
                f.write("-" * 15 + "\n")
                f.write(f"{overall['test']}: p = {overall['p_value']:.6f}\n")
                f.write(f"Significant: {'Yes' if overall['significant'] else 'No'}\n")
        
        print(f"✅ Created text summary: {summary_file}")
    
    def _export_video_frequencies(self):
        """Export individual video frequencies to CSV"""
        if self.video_frequencies is not None:
            freq_file = self.output_dir / 'individual_video_frequencies.csv'
            self.video_frequencies.to_csv(freq_file, index=False)
            print(f"✅ Exported individual video frequencies: {freq_file}")
        else:
            print("⚠️  No video frequency data to export")
    
    def _export_regions_of_interest_summary(self):
        """Export frequency sum for regions of interest per video"""
        if self.video_metadata is not None:
            # Create summary with video info and regions of interest frequency sum
            roi_summary = []
            for idx, row in self.video_metadata.iterrows():
                summary_row = {
                    'filename': row['filename'],
                    'group': row['group'],
                    'total_frames': row['total_frames'],
                    'regions_of_interest': str(self.regions_of_interest),
                    'frequency_sum_roi': row['total_frequency'],  # This is the sum of frequencies for regions of interest
                    'frequency_ratio_roi': row['total_frequency']  # Same value, but clearer naming
                }
                
                # Add individual region frequencies for the regions of interest
                if hasattr(self, 'video_frequencies') and self.video_frequencies is not None:
                    video_freq_row = self.video_frequencies[self.video_frequencies['filename'] == row['filename']]
                    if not video_freq_row.empty:
                        for region in self.regions_of_interest:
                            freq_col = f'Freq_Region_{region}'
                            if freq_col in self.video_frequencies.columns:
                                summary_row[f'freq_region_{region}'] = video_freq_row[freq_col].iloc[0]
                
                roi_summary.append(summary_row)
            
            roi_df = pd.DataFrame(roi_summary)
            roi_file = self.output_dir / 'regions_of_interest_frequencies_per_video.csv'
            roi_df.to_csv(roi_file, index=False)
            print(f"✅ Exported regions of interest frequencies per video: {roi_file}")
            print(f"   Regions of interest: {self.regions_of_interest}")
            print(f"   Videos: {len(roi_df)}")
        else:
            print("⚠️  No video metadata to export")
    
    def run_complete_analysis(self, test_type='both'):
        """Run the complete statistical analysis pipeline"""
        try:
            # Load data and calculate frequencies
            self.load_region_counts()
            self.calculate_total_frames_per_video()
            self.calculate_video_frequencies()
            self.create_video_metadata()
            self.prepare_group_data()
            
            # Statistical tests
            self.perform_pairwise_tests(test_type=test_type)
            
            # Visualizations
            self.create_visualizations()
            
            # Export results
            self.export_results()
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
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
        
        # Group rankings
        group_means = [(group, data['mean']) for group, data in self.group_data.items()]
        group_means.sort(key=lambda x: x[1], reverse=True)
        
        print("Group rankings (by mean frequency):")
        for i, (group, mean_freq) in enumerate(group_means, 1):
            print(f"  {i}. {group}: {mean_freq:.2f}")
        
        # Significant differences
        significant_pairs = [r for r in self.statistical_results['pairwise_tests'] if r.get('mannwhitney_significant', False)]
        
        if significant_pairs:
            print(f"\nSignificant differences ({len(significant_pairs)} pairs):")
            significant_pairs.sort(key=lambda x: x['mannwhitney_p'])
            for result in significant_pairs[:5]:  # Show top 5
                p_val = result['mannwhitney_p']
                d = abs(result['cohens_d'])
                print(f"  {result['group1']} vs {result['group2']}: p={p_val:.4f}, d={d:.2f}")
        else:
            print("\nNo significant differences found")
        
        # Overall test result
        if 'overall_test' in self.statistical_results:
            overall = self.statistical_results['overall_test']
            print(f"\nOverall test ({overall['test']}): p = {overall['p_value']:.6f}")
            print(f"Overall significance: {'Yes' if overall['significant'] else 'No'}")


def main():
    """Main execution function"""
    print("Starting region frequency statistical analysis...")
    
    # Initialize analyzer
    analyzer = RegionFrequencyAnalysis('analysis_outputs')
    
    # Run complete analysis
    analyzer.run_complete_analysis(test_type='both')  # Both Mann-Whitney and t-test
    
    print(f"\nResults saved to: {analyzer.output_dir}")
    print("Files created:")
    print("  - region_frequency_analysis.png (comprehensive plots)")
    print("  - pairwise_statistical_tests.csv (detailed results)")
    print("  - group_summary_statistics.csv (group summaries)")
    print("  - individual_video_frequencies.csv (frequency ratios per video)")
    print("  - regions_of_interest_frequencies_per_video.csv (ROI frequency sum per video)")
    print("  - significant_pairs.csv (significant comparisons only)")
    print("  - analysis_summary.txt (human-readable summary)")


if __name__ == "__main__":
    main()
