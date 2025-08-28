#!/usr/bin/env python3
"""
Weighted Score Analysis Based on TBI vs Other Group Differences

This script:
1. Loads region count data from MATLAB analysis outputs
2. Calculates region weights based on (Other groups - TBI) frequency differences
3. Computes weighted scores for each video: sum(frequency × weight) across all regions
4. Groups videos by experimental condition
5. Performs pairwise statistical comparisons between groups using weighted scores
6. Creates visualizations and exports results

The score emphasizes regions that differentiate TBI from other groups.

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

class WeightedScoreAnalysis:
    """Statistical analysis using weighted scores based on TBI differentiation"""
    
    def __init__(self, analysis_outputs_dir='analysis_outputs'):
        """
        Initialize the analysis
        
        Args:
            analysis_outputs_dir: Directory containing MATLAB analysis outputs
        """
        self.analysis_outputs_dir = Path(analysis_outputs_dir)
        self.csv_dir = self.analysis_outputs_dir / 'csv'
        self.output_dir = self.analysis_outputs_dir / 'weighted_score_analysis'
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.region_counts = None
        self.total_frames_per_video = None
        self.video_frequencies = None
        self.region_weights = None
        self.video_scores = None
        self.video_metadata = None
        self.group_data = None
        self.statistical_results = {}
        
        print("="*60)
        print("WEIGHTED SCORE ANALYSIS (TBI vs Other Groups)")
        print("="*60)
        print(f"Analysis outputs directory: {self.analysis_outputs_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def parse_filename_to_group(self, filename):
        """
        Parse filename to extract experimental group
        
        Args:
            filename: e.g., 'DRG_1.mat', 'week4-DRG_2.mat'
            
        Returns:
            dict with group info and TBI classification
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
                
                return {
                    'group': full_group,
                    'base_group': group,
                    'week': week,
                    'is_tbi': is_tbi,
                    'group_type': 'TBI' if is_tbi else 'Other'
                }
        
        return {
            'group': 'unknown',
            'base_group': 'unknown',
            'week': 'unknown',
            'is_tbi': False,
            'group_type': 'Other'
        }
    
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
        
        # Get all available regions
        self.all_regions = [int(col.split('_')[1]) for col in self.region_counts.columns if col.startswith('Region_')]
        self.all_regions.sort()
        print(f"   Available regions: {self.all_regions} (Total: {len(self.all_regions)})")
        
        return self.region_counts
    
    def calculate_total_frames_per_video(self):
        """Calculate total frames for each video from frame indices files"""
        print("\n2. Calculating total frames per video...")
        
        frame_indices_dir = self.csv_dir / 'frame_indices_per_video'
        if not frame_indices_dir.exists():
            # Alternative: use sum of all region counts as approximation
            print("   Frame indices directory not found, using sum of region counts")
            region_cols = [col for col in self.region_counts.columns if col.startswith('Region_')]
            total_frames = {}
            for idx, row in self.region_counts.iterrows():
                total_frames[row['File']] = row[region_cols].sum()
        else:
            total_frames = {}
            
            for idx, row in self.region_counts.iterrows():
                filename = row['File']
                # Find corresponding frame indices file
                pattern = f"*_{filename}_frame_indices.csv"
                matching_files = list(frame_indices_dir.glob(pattern))
                
                if not matching_files:
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
                    # Use sum of region counts as fallback
                    region_cols = [col for col in self.region_counts.columns if col.startswith('Region_')]
                    total_frames[filename] = row[region_cols].sum()
        
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
            for region in self.all_regions:
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
    
    def calculate_region_weights(self, fold_threshold=7):
        """Calculate binary weights: +1 if (Other / TBI) > fold_threshold, else 0"""
        print(f"\n4. Calculating binary region weights with fold change threshold = {fold_threshold:.2f}x...")
        
        # First, classify all videos
        video_classifications = []
        for idx, row in self.video_frequencies.iterrows():
            filename = row['filename']
            group_info = self.parse_filename_to_group(filename)
            video_classifications.append({
                'filename': filename,
                'is_tbi': group_info['is_tbi'],
                'group_type': group_info['group_type']
            })
        
        classifications_df = pd.DataFrame(video_classifications)
        
        # Calculate binary weights for each region based on threshold
        self.region_weights = {}
        region_stats = []
        
        for region in self.all_regions:
            freq_col = f'Freq_Region_{region}'
            if freq_col in self.video_frequencies.columns:
                # Merge frequencies with classifications
                freq_data = self.video_frequencies[['filename', freq_col]].merge(
                    classifications_df, on='filename'
                )
                
                # Calculate means for each group type
                tbi_mean = freq_data[freq_data['is_tbi']]['Freq_Region_{}'.format(region)].mean()
                other_mean = freq_data[~freq_data['is_tbi']]['Freq_Region_{}'.format(region)].mean()
                
                # Add small pseudocount to avoid division by zero
                pseudocount = 1e-6
                tbi_mean_adj = tbi_mean + pseudocount
                other_mean_adj = other_mean + pseudocount
                
                # Calculate fold change
                fold_change = other_mean_adj / tbi_mean_adj
                raw_diff = other_mean - tbi_mean  # Keep for reference
                
                # BINARY WEIGHT: 1 if fold change > threshold, else 0
                weight = 1 if fold_change > fold_threshold else 0
                
                self.region_weights[region] = weight
                
                region_stats.append({
                    'Region': region,
                    'TBI_Mean': tbi_mean,
                    'Other_Mean': other_mean,
                    'Raw_Difference': raw_diff,
                    'Fold_Change': fold_change,
                    'Above_Threshold': fold_change > fold_threshold,
                    'Weight': weight,
                    'Included': 'Yes' if weight > 0 else 'No'
                })
                
                status = "INCLUDED (+1)" if weight > 0 else "EXCLUDED (0)"
                print(f"   Region {region}: TBI={tbi_mean:.4f}, Other={other_mean:.4f}, FC={fold_change:.2f}x, Weight={weight} [{status}]")
        
        # Save region weights for reference
        weights_df = pd.DataFrame(region_stats)
        weights_df = weights_df.sort_values(['Weight', 'Fold_Change'], ascending=[False, False])
        weights_file = self.output_dir / 'region_weights_tbi_differentiation.csv'
        weights_df.to_csv(weights_file, index=False)
        print(f"\n✅ Saved region weights to: {weights_file}")
        
        # Summary statistics
        included_regions = sum(1 for w in self.region_weights.values() if w > 0)
        excluded_regions = sum(1 for w in self.region_weights.values() if w == 0)
        
        print(f"\n📊 Binary Weight Summary (Fold Change Threshold = {fold_threshold:.2f}x):")
        print(f"   Regions included (weight = 1): {included_regions}")
        print(f"   Regions excluded (weight = 0): {excluded_regions}")
        print(f"   Total regions: {len(self.region_weights)}")
        print(f"   Max possible score: {included_regions}")
        
        # Show included regions
        included_regions_list = [(r, region_stats[i]['Fold_Change']) for i, (r, w) in enumerate(self.region_weights.items()) if w > 0]
        included_regions_list.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔝 Included regions (sorted by fold change):")
        for i, (region, fc) in enumerate(included_regions_list, 1):
            print(f"   {i}. Region {region}: fold change = {fc:.2f}x")
        
        return self.region_weights
    
    def calculate_weighted_scores(self):
        """Calculate syndrome score for each video: count of regions above threshold"""
        print("\n5. Calculating syndrome scores for each video (count of active regions)...")
        
        scores_data = []
        
        for idx, row in self.video_frequencies.iterrows():
            filename = row['filename']
            group_info = self.parse_filename_to_group(filename)
            
            # Calculate syndrome score: sum frequencies for regions above threshold
            score = 0
            region_contributions = {}
            
            for region in self.all_regions:
                freq_col = f'Freq_Region_{region}'
                if freq_col in self.video_frequencies.columns and region in self.region_weights:
                    frequency = row[freq_col]
                    # Only include regions that passed the threshold (weight = 1)
                    if self.region_weights[region] == 1:
                        # Add the actual frequency for this region
                        contribution = frequency
                        score += contribution
                        region_contributions[f'Region_{region}_freq'] = contribution
                    else:
                        # Region was excluded, doesn't contribute
                        region_contributions[f'Region_{region}_freq'] = 0
            
            score_row = {
                'filename': filename,
                'group': group_info['group'],
                'is_tbi': group_info['is_tbi'],
                'group_type': group_info['group_type'],
                'syndrome_score': score,
                'num_included_regions': sum(1 for w in self.region_weights.values() if w == 1),
                'total_frames': row['total_frames'],
                **region_contributions  # Add individual region contributions
            }
            
            scores_data.append(score_row)
        
        self.video_scores = pd.DataFrame(scores_data)
        
        # Save detailed scores
        scores_file = self.output_dir / 'video_syndrome_scores_detailed.csv'
        self.video_scores.to_csv(scores_file, index=False)
        print(f"✅ Saved detailed video scores to: {scores_file}")
        
        # Create summary version
        summary_scores = self.video_scores[['filename', 'group', 'group_type', 'syndrome_score', 'num_included_regions', 'total_frames']]
        summary_file = self.output_dir / 'video_syndrome_scores_summary.csv'
        summary_scores.to_csv(summary_file, index=False)
        print(f"✅ Saved summary video scores to: {summary_file}")
        
        # Display group means
        print("\n📊 Syndrome Score Summary by Group:")
        group_stats = self.video_scores.groupby('group').agg({
            'syndrome_score': ['mean', 'std', 'count', 'min', 'max']
        }).round(4)
        print(group_stats)
        
        return self.video_scores
    
    def prepare_group_data(self):
        """Prepare data grouped by experimental condition"""
        print("\n6. Preparing group data for statistical analysis...")
        
        # Group data by experimental condition
        self.group_data = {}
        
        for group in self.video_scores['group'].unique():
            if group == 'unknown':
                continue
                
            group_videos = self.video_scores[self.video_scores['group'] == group]
            scores = group_videos['syndrome_score'].values
            
            self.group_data[group] = {
                'scores': scores,
                'n_videos': len(scores),
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'sem': np.std(scores) / np.sqrt(len(scores)) if len(scores) > 0 else 0
            }
            
            print(f"   {group}: {len(scores)} videos, mean={np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        return self.group_data
    
    def perform_pairwise_tests(self, test_type='both'):
        """
        Perform pairwise statistical tests between groups WITHIN THE SAME WEEK ONLY
        
        Args:
            test_type: 'mannwhitney', 'ttest', or 'both'
        """
        print(f"\n7. Performing within-week pairwise statistical tests ({test_type})...")
        
        groups = list(self.group_data.keys())
        n_groups = len(groups)
        
        if n_groups < 2:
            print("❌ Need at least 2 groups for pairwise testing")
            return
        
        # Separate groups by week
        week1_groups = [g for g in groups if not g.startswith('week4-')]
        week4_groups = [g for g in groups if g.startswith('week4-')]
        
        print(f"   Week 1 groups: {week1_groups}")
        print(f"   Week 4 groups: {week4_groups}")
        
        # Initialize results storage
        self.statistical_results = {
            'pairwise_tests': [],
            'summary': {
                'n_groups': n_groups,
                'groups': groups,
                'test_type': test_type,
                'score_type': 'weighted_tbi_differentiation',
                'comparison_type': 'within_week_only'
            }
        }
        
        # Count total within-week comparisons
        week1_comparisons = len(week1_groups) * (len(week1_groups) - 1) // 2
        week4_comparisons = len(week4_groups) * (len(week4_groups) - 1) // 2
        total_comparisons = week1_comparisons + week4_comparisons
        print(f"   Total within-week comparisons: {total_comparisons} (Week1: {week1_comparisons}, Week4: {week4_comparisons})")
        
        # Perform pairwise comparisons WITHIN each week
        # Week 1 comparisons
        for i, group1 in enumerate(week1_groups):
            for j, group2 in enumerate(week1_groups[i+1:], i+1):
                
                data1 = self.group_data[group1]['scores']
                data2 = self.group_data[group2]['scores']
                
                # Prepare result dict
                result = {
                    'group1': group1,
                    'group2': group2,
                    'n1': len(data1),
                    'n2': len(data2),
                    'mean1': np.mean(data1),
                    'mean2': np.mean(data2),
                    'std1': np.std(data1),
                    'std2': np.std(data2),
                    'mean_difference': np.mean(data1) - np.mean(data2)
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
        
        # Week 4 comparisons
        for i, group1 in enumerate(week4_groups):
            for j, group2 in enumerate(week4_groups[i+1:], i+1):
                
                data1 = self.group_data[group1]['scores']
                data2 = self.group_data[group2]['scores']
                
                # Prepare result dict
                result = {
                    'group1': group1,
                    'group2': group2,
                    'n1': len(data1),
                    'n2': len(data2),
                    'mean1': np.mean(data1),
                    'mean2': np.mean(data2),
                    'std1': np.std(data1),
                    'std2': np.std(data2),
                    'mean_difference': np.mean(data1) - np.mean(data2)
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
        
        # Overall test (Kruskal-Wallis) WITHIN each week
        # Week 1 overall test
        if len(week1_groups) > 2:
            try:
                week1_data = [self.group_data[group]['scores'] for group in week1_groups]
                kw_statistic, kw_p = kruskal(*week1_data)
                self.statistical_results['week1_overall_test'] = {
                    'test': 'Kruskal-Wallis (Week 1)',
                    'groups': week1_groups,
                    'statistic': kw_statistic,
                    'p_value': kw_p,
                    'significant': kw_p < 0.05
                }
                print(f"   Week 1 Kruskal-Wallis test: p = {kw_p:.6f}")
            except Exception as e:
                print(f"   Warning: Week 1 Kruskal-Wallis test failed: {e}")
        
        # Week 4 overall test
        if len(week4_groups) > 2:
            try:
                week4_data = [self.group_data[group]['scores'] for group in week4_groups]
                kw_statistic, kw_p = kruskal(*week4_data)
                self.statistical_results['week4_overall_test'] = {
                    'test': 'Kruskal-Wallis (Week 4)',
                    'groups': week4_groups,
                    'statistic': kw_statistic,
                    'p_value': kw_p,
                    'significant': kw_p < 0.05
                }
                print(f"   Week 4 Kruskal-Wallis test: p = {kw_p:.6f}")
            except Exception as e:
                print(f"   Warning: Week 4 Kruskal-Wallis test failed: {e}")
        
        # Save pairwise results
        results_df = pd.DataFrame(self.statistical_results['pairwise_tests'])
        results_file = self.output_dir / 'pairwise_statistical_tests_weighted_scores.csv'
        results_df.to_csv(results_file, index=False)
        print(f"✅ Saved pairwise test results to: {results_file}")
        
        # Summary statistics
        if test_type in ['mannwhitney', 'both']:
            significant_mw = sum(1 for r in self.statistical_results['pairwise_tests'] if r.get('mannwhitney_significant', False))
            print(f"   Mann-Whitney significant pairs: {significant_mw}/{len(self.statistical_results['pairwise_tests'])}")
        
        if test_type in ['ttest', 'both']:
            significant_tt = sum(1 for r in self.statistical_results['pairwise_tests'] if r.get('ttest_significant', False))
            print(f"   T-test significant pairs: {significant_tt}/{len(self.statistical_results['pairwise_tests'])}")
        
        return self.statistical_results
    
    def _add_significance_bracket(self, ax, x1, x2, y, p_value, height_factor=0.02):
        """Add significance bracket between two groups on a plot"""
        # Convert p-value to stars
        if p_value < 0.001:
            sig_text = '***'
        elif p_value < 0.01:
            sig_text = '**'
        elif p_value < 0.05:
            sig_text = '*'
        else:
            return  # Don't show non-significant comparisons
        
        # Get y-axis range for scaling
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        bracket_height = y + height_factor * y_range
        
        # Draw bracket
        ax.plot([x1, x1, x2, x2], 
                [y, bracket_height, bracket_height, y], 
                'k-', linewidth=1)
        
        # Add significance text
        ax.text((x1 + x2) / 2, bracket_height + 0.005 * y_range, 
                sig_text, ha='center', va='bottom', fontsize=10)
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n8. Creating visualizations...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Syndrome Score Analysis (Fold Change Threshold)', fontsize=16, fontweight='bold')
        
        # 1. Box plot by group with within-week t-test significance
        ax1 = axes[0, 0]
        groups = list(self.group_data.keys())
        scores = [self.group_data[group]['scores'] for group in groups]
        
        # Separate groups by week
        week1_groups = [g for g in groups if not g.startswith('week4-')]
        week4_groups = [g for g in groups if g.startswith('week4-')]
        
        # Create boxplot
        bp = ax1.boxplot(scores, labels=groups, patch_artist=True)
        
        # Color boxes by week
        colors_by_week = []
        for g in groups:
            if g in week1_groups:
                colors_by_week.append('lightblue')
            else:
                colors_by_week.append('lightcoral')
        
        for patch, color in zip(bp['boxes'], colors_by_week):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add within-week significance annotations
        # First, find max values for each group to position brackets
        max_values = []
        for score_list in scores:
            if len(score_list) > 0:
                # Use 75th percentile + 1.5*IQR as reference (similar to boxplot whiskers)
                q75 = np.percentile(score_list, 75)
                q25 = np.percentile(score_list, 25)
                iqr = q75 - q25
                max_val = min(q75 + 1.5 * iqr, max(score_list))
                max_values.append(max_val)
            else:
                max_values.append(0)
        
        # Week 1 comparisons
        week1_indices = [i for i, g in enumerate(groups) if g in week1_groups]
        bracket_level = 0
        for i, idx1 in enumerate(week1_indices):
            for idx2 in week1_indices[i+1:]:
                # Get t-test p-value for this pair
                group1 = groups[idx1]
                group2 = groups[idx2]
                
                # Find the result in our statistical tests
                for result in self.statistical_results['pairwise_tests']:
                    if ((result['group1'] == group1 and result['group2'] == group2) or 
                        (result['group1'] == group2 and result['group2'] == group1)):
                        p_val = result.get('ttest_p', 1.0)
                        if p_val < 0.05:  # Only show significant comparisons
                            # Position bracket above the higher group
                            bracket_y = max(max_values[idx1], max_values[idx2]) + bracket_level * 0.08 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
                            self._add_significance_bracket(ax1, idx1+1, idx2+1, bracket_y, p_val)
                            bracket_level += 1
                        break
        
        # Week 4 comparisons
        week4_indices = [i for i, g in enumerate(groups) if g in week4_groups]
        bracket_level = 0
        for i, idx1 in enumerate(week4_indices):
            for idx2 in week4_indices[i+1:]:
                # Get t-test p-value for this pair
                group1 = groups[idx1]
                group2 = groups[idx2]
                
                # Find the result in our statistical tests
                for result in self.statistical_results['pairwise_tests']:
                    if ((result['group1'] == group1 and result['group2'] == group2) or 
                        (result['group1'] == group2 and result['group2'] == group1)):
                        p_val = result.get('ttest_p', 1.0)
                        if p_val < 0.05:  # Only show significant comparisons
                            # Position bracket above the higher group
                            bracket_y = max(max_values[idx1], max_values[idx2]) + bracket_level * 0.08 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
                            self._add_significance_bracket(ax1, idx1+1, idx2+1, bracket_y, p_val)
                            bracket_level += 1
                        break
        
        ax1.set_ylabel('Syndrome Score (Sum of Frequencies)')
        ax1.set_title('Syndrome Score Distribution by Group\n(Blue=Week1, Red=Week4, *p<0.05, **p<0.01, ***p<0.001)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No syndrome activity')
        
        # Adjust y-axis limits to accommodate brackets
        current_ylim = ax1.get_ylim()
        ax1.set_ylim(current_ylim[0], current_ylim[1] * 1.3)
        ax1.legend()
        
        # 2. Violin plot
        ax2 = axes[0, 1]
        
        # Prepare data for seaborn
        plot_data = []
        for group in groups:
            for score in self.group_data[group]['scores']:
                plot_data.append({'Group': group, 'Score': score})
        plot_df = pd.DataFrame(plot_data)
        
        sns.violinplot(data=plot_df, x='Group', y='Score', ax=ax2)
        ax2.set_title('Syndrome Score Density by Group')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # 3. Region weights visualization
        ax3 = axes[0, 2]
        weights = list(self.region_weights.values())
        regions = list(self.region_weights.keys())
        colors_weights = ['blue' if w > 0 else 'gray' for w in weights]  # Gray for zero weights
        
        bars = ax3.bar(range(len(weights)), weights, color=colors_weights, alpha=0.7)
        ax3.set_xlabel('Region')
        ax3.set_ylabel('Binary Weight (0 or 1)')
        ax3.set_title('Region Inclusion for Syndrome Score\n(Blue = Included, Gray = Excluded)')
        ax3.set_xticks(range(len(regions)))
        ax3.set_xticklabels(regions, rotation=45)
        ax3.set_ylim(-0.1, 1.1)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.axhline(y=1, color='blue', linestyle='--', alpha=0.5, label='Included regions')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Statistical significance heatmap
        ax4 = axes[1, 0]
        
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
        im = ax4.imshow(p_matrix, cmap='RdYlBu_r', vmin=0, vmax=0.1)
        ax4.set_xticks(range(n_groups))
        ax4.set_yticks(range(n_groups))
        ax4.set_xticklabels(groups, rotation=45)
        ax4.set_yticklabels(groups)
        ax4.set_title('P-values (Mann-Whitney U)')
        
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
                    ax4.text(j, i, text, ha='center', va='center', 
                            color='white' if p_val < 0.05 else 'black', fontweight='bold')
        
        plt.colorbar(im, ax=ax4, label='p-value')
        
        # 5. Effect sizes
        ax5 = axes[1, 1]
        
        comparison_names = []
        effect_sizes = []
        p_values = []
        
        for result in self.statistical_results['pairwise_tests']:
            comp_name = f"{result['group1']}\nvs\n{result['group2']}"
            comparison_names.append(comp_name)
            effect_sizes.append(abs(result['cohens_d']))
            p_values.append(result.get('mannwhitney_p', 1.0))
        
        colors_effect = ['red' if p < 0.05 else 'gray' for p in p_values]
        bars = ax5.bar(range(len(effect_sizes)), effect_sizes, color=colors_effect, alpha=0.7)
        
        ax5.set_ylabel("Cohen's d (Effect Size)")
        ax5.set_title('Effect Sizes (Red = Significant)')
        ax5.set_xticks(range(len(comparison_names)))
        ax5.set_xticklabels(comparison_names, rotation=45, ha='right')
        ax5.grid(True, alpha=0.3)
        
        # Add horizontal line at effect size thresholds
        ax5.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Small effect')
        ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Medium effect')
        ax5.axhline(y=0.8, color='darkred', linestyle='--', alpha=0.5, label='Large effect')
        ax5.legend()
        
        # 6. Group means comparison
        ax6 = axes[1, 2]
        
        # Separate TBI and Other groups
        tbi_groups = [g for g in groups if 'TBI' in g]
        other_groups = [g for g in groups if 'TBI' not in g]
        
        positions = []
        means = []
        errors = []
        colors_groups = []
        
        for i, group in enumerate(other_groups + tbi_groups):
            positions.append(i)
            means.append(self.group_data[group]['mean'])
            errors.append(self.group_data[group]['sem'])
            colors_groups.append('red' if 'TBI' in group else 'blue')
        
        ax6.bar(positions, means, yerr=errors, color=colors_groups, alpha=0.7, capsize=5)
        ax6.set_ylabel('Mean Syndrome Score ± SEM')
        ax6.set_title('Group Means (Blue=Syndrome, Red=Control/TBI)')
        ax6.set_xticks(positions)
        ax6.set_xticklabels(other_groups + tbi_groups, rotation=45, ha='right')
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / 'syndrome_score_analysis_comprehensive.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comprehensive plot: {fig_path}")
        
        plt.show()
    
    def export_summary_results(self):
        """Export summary results and create text report"""
        print("\n9. Exporting summary results...")
        
        # Group summary
        group_summary = []
        for group, data in self.group_data.items():
            group_summary.append({
                'Group': group,
                'N_Videos': data['n_videos'],
                'Mean_Score': data['mean'],
                'Std_Score': data['std'],
                'Median_Score': data['median'],
                'SEM': data['sem']
            })
        
        summary_df = pd.DataFrame(group_summary)
        summary_file = self.output_dir / 'group_summary_weighted_scores.csv'
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Exported group summary: {summary_file}")
        
        # Significant pairs
        significant_pairs = []
        for result in self.statistical_results['pairwise_tests']:
            if result.get('mannwhitney_significant', False):
                significant_pairs.append({
                    'Group1': result['group1'],
                    'Group2': result['group2'],
                    'P_value': result['mannwhitney_p'],
                    'Effect_Size': abs(result['cohens_d']),
                    'Mean1': result['mean1'],
                    'Mean2': result['mean2'],
                    'Mean_Difference': result['mean_difference']
                })
        
        if significant_pairs:
            sig_df = pd.DataFrame(significant_pairs)
            sig_file = self.output_dir / 'significant_pairs_weighted_scores.csv'
            sig_df.to_csv(sig_file, index=False)
            print(f"✅ Exported significant pairs: {sig_file}")
        
        # Create text summary
        self._create_text_summary()
    
    def _create_text_summary(self):
        """Create a human-readable text summary"""
        summary_file = self.output_dir / 'weighted_score_analysis_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("SYNDROME SCORE ANALYSIS SUMMARY (FOLD CHANGE THRESHOLD)\n")
            f.write("=" * 55 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Regions Used: {len(self.all_regions)}\n")
            f.write(f"Total Videos: {len(self.video_scores)}\n\n")
            
            # Weight summary
            f.write("REGION INCLUSION SUMMARY (FOLD CHANGE THRESHOLD):\n")
            f.write("-" * 50 + "\n")
            included_regions = sum(1 for w in self.region_weights.values() if w == 1)
            excluded_regions = sum(1 for w in self.region_weights.values() if w == 0)
            f.write(f"Regions included (weight = 1): {included_regions}\n")
            f.write(f"Regions excluded (weight = 0): {excluded_regions}\n")
            f.write(f"Maximum possible syndrome score: {included_regions}\n")
            
            # Included regions
            included_regions_list = [r for r, w in self.region_weights.items() if w == 1]
            f.write(f"\nIncluded regions (contribute to syndrome score):\n")
            for region in included_regions_list:
                f.write(f"  Region {region}\n")
            
            # Group statistics
            f.write("\n\nGROUP STATISTICS:\n")
            f.write("-" * 30 + "\n")
            for group, data in self.group_data.items():
                f.write(f"{group}: {data['n_videos']} videos\n")
                f.write(f"  Mean ± SEM: {data['mean']:.4f} ± {data['sem']:.4f}\n")
                f.write(f"  Median: {data['median']:.4f}\n")
                f.write(f"  Range: {np.min(data['scores']):.4f} - {np.max(data['scores']):.4f}\n\n")
            
            # Significant comparisons
            f.write("SIGNIFICANT PAIRWISE COMPARISONS (p < 0.05):\n")
            f.write("-" * 50 + "\n")
            
            significant_count = 0
            for result in self.statistical_results['pairwise_tests']:
                if result.get('mannwhitney_significant', False):
                    significant_count += 1
                    f.write(f"{result['group1']} vs {result['group2']}:\n")
                    f.write(f"  p = {result['mannwhitney_p']:.6f}\n")
                    f.write(f"  Effect size (Cohen's d) = {abs(result['cohens_d']):.3f}\n")
                    f.write(f"  Mean difference = {result['mean_difference']:.4f}\n")
                    f.write(f"  {result['group1']}: {result['mean1']:.4f} ± {result['std1']:.4f}\n")
                    f.write(f"  {result['group2']}: {result['mean2']:.4f} ± {result['std2']:.4f}\n\n")
            
            if significant_count == 0:
                f.write("No significant differences found.\n\n")
            else:
                f.write(f"Total significant pairs: {significant_count}\n\n")
            
            # Overall tests
            f.write("OVERALL TESTS:\n")
            f.write("-" * 20 + "\n")
            
            if 'week1_overall_test' in self.statistical_results:
                week1_overall = self.statistical_results['week1_overall_test']
                f.write(f"{week1_overall['test']}: p = {week1_overall['p_value']:.6f}\n")
                f.write(f"Week 1 Significant: {'Yes' if week1_overall['significant'] else 'No'}\n\n")
            
            if 'week4_overall_test' in self.statistical_results:
                week4_overall = self.statistical_results['week4_overall_test']
                f.write(f"{week4_overall['test']}: p = {week4_overall['p_value']:.6f}\n")
                f.write(f"Week 4 Significant: {'Yes' if week4_overall['significant'] else 'No'}\n")
        
        print(f"✅ Created text summary: {summary_file}")
    
    def run_complete_analysis(self, test_type='both'):
        """Run the complete weighted score analysis pipeline"""
        try:
            # Load data and calculate frequencies
            self.load_region_counts()
            self.calculate_total_frames_per_video()
            self.calculate_video_frequencies()
            
            # Calculate weights and scores
            self.calculate_region_weights()
            self.calculate_weighted_scores()
            
            # Statistical analysis
            self.prepare_group_data()
            self.perform_pairwise_tests(test_type=test_type)
            
            # Visualizations and export
            self.create_visualizations()
            self.export_summary_results()
            
            print("\n" + "="*60)
            print("WEIGHTED SCORE ANALYSIS COMPLETE")
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
        
        # Group rankings by weighted score
        group_means = [(group, data['mean']) for group, data in self.group_data.items()]
        group_means.sort(key=lambda x: x[1], reverse=True)
        
        print("\nGroup rankings (by mean syndrome score):")
        for i, (group, mean_score) in enumerate(group_means, 1):
            print(f"  {i}. {group}: {mean_score:.4f}")
        
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
        
        # Overall test results
        if 'week1_overall_test' in self.statistical_results:
            week1_overall = self.statistical_results['week1_overall_test']
            print(f"\nWeek 1 Kruskal-Wallis test: p = {week1_overall['p_value']:.6f}")
            print(f"Week 1 significance: {'Yes' if week1_overall['significant'] else 'No'}")
        
        if 'week4_overall_test' in self.statistical_results:
            week4_overall = self.statistical_results['week4_overall_test']
            print(f"\nWeek 4 Kruskal-Wallis test: p = {week4_overall['p_value']:.6f}")
            print(f"Week 4 significance: {'Yes' if week4_overall['significant'] else 'No'}")


def main():
    """Main execution function"""
    print("Starting syndrome score analysis using fold change threshold approach...")
    
    # Initialize analyzer
    analyzer = WeightedScoreAnalysis('analysis_outputs')
    
    # Run complete analysis
    analyzer.run_complete_analysis(test_type='both')  # Both Mann-Whitney and t-test
    
    print(f"\nResults saved to: {analyzer.output_dir}")
    print("Files created:")
    print("  - region_weights_tbi_differentiation.csv (fold change based region inclusion)")
    print("  - video_syndrome_scores_detailed.csv (scores with region contributions)")
    print("  - video_syndrome_scores_summary.csv (summary scores)")
    print("  - pairwise_statistical_tests_weighted_scores.csv (statistical results)")
    print("  - group_summary_weighted_scores.csv (group statistics)")
    print("  - significant_pairs_weighted_scores.csv (significant comparisons)")
    print("  - syndrome_score_analysis_comprehensive.png (visualizations)")
    print("  - weighted_score_analysis_summary.txt (text summary)")


if __name__ == "__main__":
    main()
