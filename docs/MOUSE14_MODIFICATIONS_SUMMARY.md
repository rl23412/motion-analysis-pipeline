# Mouse14 Format Modifications Summary

## Overview

I've successfully modified your custom embedding pipeline to work with the **14-joint mouse14 format** instead of the original 23-joint rat23 format. The modifications include updating both the main embedding pipeline and creating a new version of the flip analysis script.

## Files Modified/Created

### 1. `custom_embedding_pipeline_SNI_TBI.m` - ✅ UPDATED
**Key Changes:**
- **Joint count**: Changed from 23 joints to 14 joints
- **Joint mapping for flipping**: Updated for mouse14 left-right pairs
- **Skeleton definition**: New connections based on your mouse14 structure
- **Distance calculations**: Updated joint references
- **Validation**: All dimensions now expect 14 joints

### 2. `analyze_flip_differences_mouse14.m` - ✅ NEW FILE
**Features:**
- Completely rewritten for mouse14 format
- Works with your SNI_2 + week4-TBI_3 training results
- Loads data from `/work/rl349/dannce/mouse14/allData/`
- Includes biomechanical analysis for 14-joint skeleton
- Creates comprehensive visualizations and CSV exports

## Mouse14 Joint Structure Implementation

### Joint Mapping (1-indexed for MATLAB)
```
1:  Snout
2:  EarL  
3:  EarR
4:  SpineF
5:  SpineM
6:  Tail(base)
7:  ForShdL
8:  ForepawL
9:  ForeShdR
10: ForepawR
11: HindShdL
12: HindpawL
13: HindShdR  
14: HindpawR
```

### Skeleton Connections
```matlab
joints_idx = [
    1 2; 1 3; 2 3;          % head: Snout-EarL, Snout-EarR, EarL-EarR
    1 4; 4 5; 5 6;          % spine: Snout-SpineF, SpineF-SpineM, SpineM-Tail(base)
    4 7; 7 8;               % left front: SpineF-ForShdL, ForShdL-ForepawL
    4 9; 9 10;              % right front: SpineF-ForeShdR, ForeShdR-ForepawR
    5 11; 11 12;            % left hind: SpineM-HindShdL, HindShdL-HindpawL
    5 13; 13 14             % right hind: SpineM-HindShdR, HindShdR-HindpawR
];
```

### Left-Right Joint Pairs for Flipping
```matlab
jointMapping = 1:14;  % Initialize with identity
jointMapping([2, 3]) = [3, 2];      % EarL <-> EarR
jointMapping([7, 9]) = [9, 7];      % ForShdL <-> ForeShdR  
jointMapping([8, 10]) = [10, 8];    % ForepawL <-> ForepawR
jointMapping([11, 13]) = [13, 11];  % HindShdL <-> HindShdR
jointMapping([12, 14]) = [14, 12];  % HindpawL <-> HindpawR
% Center joints (1, 4, 5, 6) stay the same
```

## Key Technical Updates

### 1. Distance Calculations
- **Before**: Used joints 1 and 7 (nose to tail base) 
- **After**: Uses joints 1 and 6 (snout to tail base)

### 2. Floor Value Calculation  
- **Before**: Used joints [19, 23] (hind paws in rat23)
- **After**: Uses joints [12, 14] (hind paws in mouse14)

### 3. Joint Pair Calculations
- **Before**: 23×23 = 529 pairwise distances
- **After**: 14×14 = 196 pairwise distances (more efficient!)

### 4. Skeleton Visualization
- Updated color mapping for 14 connections instead of 23
- Proper body part coloring: head (orange), spine (green), front limbs (blue/red), hind limbs (cyan/magenta)

## New Biomechanical Analysis Features

### Front Limb Asymmetry
- Calculates angles: SpineF → ForShdL/R → ForepawL/R
- Measures left vs right front limb angle differences

### Hind Limb Asymmetry  
- Calculates angles: SpineM → HindShdL/R → HindpawL/R
- Measures left vs right hind limb angle differences

### Comprehensive Metrics
- Centroid distance between original/flipped embeddings
- Density overlap (Jaccard similarity)
- Average displacement per frame
- Front/hind limb asymmetries
- Overall biomechanical asymmetry

## Usage Instructions

### Step 1: Run Main Pipeline
```matlab
% This now works with 14-joint mouse14 format
custom_embedding_pipeline_SNI_TBI
```

### Step 2: Run Flip Analysis  
```matlab
% Use the new mouse14-specific version
analyze_flip_differences_mouse14
```

## Expected Data Format

Your `.mat` files should contain a `pred` field with dimensions:
- **Shape**: `[n_frames, 3, 14]`
- **Coordinates**: `[x, y, z]` for each of 14 joints
- **Joints**: Ordered as specified in the mouse14 joint mapping above

## Output Files

### From Main Pipeline:
- `complete_embedding_results_SNI_TBI.mat` - Complete results with 14-joint embeddings
- Training density maps and watershed regions
- Group comparison visualizations

### From Flip Analysis:
- `flip_analysis_output/csv_results/flip_analysis_results_mouse14.csv` - Detailed results
- `flip_analysis_output/csv_results/group_summary_statistics_mouse14.csv` - Group summaries  
- `flip_analysis_output/plots/` - Comprehensive visualizations
- Individual mouse density maps
- Biomechanical asymmetry analyses

## Validation

### Data Requirements:
✅ Files in `/work/rl349/dannce/mouse14/allData/`  
✅ Each `.mat` file contains `pred` field  
✅ Data shape: `[n_frames, 3, 14]` (not 23!)  
✅ Training files: `SNI_2.mat` and `week4-TBI_3.mat`

### File List (30 total files):
**Training Files (2):**
- `SNI_2.mat`
- `week4-TBI_3.mat`

**Re-embedding Files (28):**
- DRG: `DRG_1.mat` through `DRG_5.mat`
- IT: `IT_1.mat`, `IT_2.mat`  
- SC: `SC_1.mat` through `SC_6.mat`
- SNI: `SNI_1.mat`, `SNI_3.mat`
- Week4 files: `week4-DRG_1.mat` through `week4-TBI_4.mat`

## Key Advantages of Mouse14 Format

### Performance Benefits:
- **Faster processing**: 196 vs 529 pairwise distance calculations  
- **Lower memory**: ~60% reduction in joint pair computations
- **Cleaner skeleton**: More anatomically relevant connections

### Analysis Benefits:
- **Better asymmetry detection**: Focused on key limb joints
- **Cleaner biomechanics**: Simplified but meaningful angle calculations
- **Targeted analysis**: Front/hind limb separation

## Troubleshooting

### Common Issues:

**Issue**: "Size mismatch - expected 14 joints, found 23"
```
Solution: Verify your data files contain 14-joint mouse14 format, not 23-joint rat23
Check: load('your_file.mat'); size(pred)  % Should be [n_frames, 3, 14]
```

**Issue**: "Joint index out of bounds"  
```
Solution: Ensure all joint references use 1-14, not 1-23
The modifications should handle this automatically
```

**Issue**: "Missing embedding results"
```
Solution: Run custom_embedding_pipeline_SNI_TBI.m first
This creates complete_embedding_results_SNI_TBI.mat needed for flip analysis
```

## Testing Recommendations

### Before Running:
1. **Verify data format**: Check a sample file has 14 joints
2. **Check file paths**: Ensure `/work/rl349/dannce/mouse14/allData/` exists
3. **Validate training files**: Confirm `SNI_2.mat` and `week4-TBI_3.mat` are present

### After Running:
1. **Check outputs**: Verify `complete_embedding_results_SNI_TBI.mat` is created
2. **Validate plots**: Ensure skeleton looks correct in visualizations  
3. **Review metrics**: Check that asymmetry values are reasonable (typically 0-180 degrees)

## Summary

✅ **Complete conversion to mouse14 format**  
✅ **Updated skeleton with 14 joints and appropriate connections**  
✅ **Proper left-right joint mapping for asymmetry analysis**  
✅ **Enhanced biomechanical analysis with front/hind limb metrics**  
✅ **Full compatibility with your SNI_2 + week4-TBI_3 training approach**  
✅ **Comprehensive visualization and CSV export capabilities**  
✅ **Performance optimizations due to fewer joints**

The system is now ready to work with your 14-joint mouse14 data format while maintaining all the advanced embedding and analysis capabilities you need.

---

**Files Ready to Use:**
- `custom_embedding_pipeline_SNI_TBI.m` (updated for mouse14)
- `analyze_flip_differences_mouse14.m` (new mouse14-specific version)

**Next Step:** Run the main pipeline with your mouse14 data!






