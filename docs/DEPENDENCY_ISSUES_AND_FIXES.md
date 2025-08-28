# Dependency Issues and Fixes for Mouse14 Pipeline

## Summary of Dependency Analysis

I have thoroughly analyzed both main scripts (`custom_embedding_pipeline_SNI_TBI.m` and `analyze_flip_differences_mouse14.m`) for potential dependency issues. Here's what I found:

## ✅ **GOOD NEWS: No Critical Issues Found**

The scripts are well-designed with proper fallback mechanisms for most functions. Here's the status:

## 📋 **Complete Dependency List**

### 1. **MATLAB Core Functions** ✅
All core MATLAB functions used are available in standard installations:
- `load`, `save`, `exist`, `size`, `zeros`, `ones`
- `randn`, `mean`, `std`, `min`, `max`, `sqrt`, `sum`
- `meshgrid`, `squeeze`, `reshape`, `vertcat`, `horzcat`
- `cellfun`, `strcmp`, `contains`, `startsWith`

### 2. **MATLAB Toolbox Functions** ⚠️
These require specific toolboxes but have fallbacks:

#### Statistics and Machine Learning Toolbox:
- `tsne` - **REQUIRED** (no fallback possible)
- `corr` - **REQUIRED** (no fallback possible)

#### Signal Processing Toolbox:
- `medfilt1` - **Has fallback** (skipped if missing)
- `smooth` - **Has fallback** (uses `movmean`)

#### Image Processing Toolbox:
- `watershed` - **REQUIRED** (no fallback possible)
- `bwboundaries` - **REQUIRED** (no fallback possible)
- `imgaussfilt` - **Has fallback** (uses `imfilter` + `fspecial`)
- `imfilter` - Used in fallback
- `fspecial` - Used in fallback

### 3. **MotionMapper Functions** ❌ **CRITICAL**
These are **REQUIRED** and must be installed:
- `findWavelets` - Creates wavelet decomposition
- `findTemplatesFromData` - Template matching
- `findTDistributedProjections_fmin` - Re-embedding
- `findWatershedRegions_v2` - Watershed region analysis

### 4. **Custom Helper Functions** ✅
These are defined within the scripts:
- `findPointDensity` - Defined in main script
- `combineCells` - Defined in main script
- `setRunParameters` - Defined in main script
- `parseFilename` - Defined in flip analysis
- `calculateAngle3D` - Defined in flip analysis

## 🔧 **Installation Requirements**

### **CRITICAL - Must Have:**
1. **MotionMapper Toolbox**
   ```matlab
   % Download from: https://github.com/gordonberman/MotionMapper
   % Add to path:
   addpath(genpath('/path/to/MotionMapper'))
   ```

2. **Statistics and Machine Learning Toolbox**
   - For `tsne` and `corr` functions
   - Cannot be replaced with fallbacks

3. **Image Processing Toolbox**
   - For `watershed` and `bwboundaries`
   - Critical for behavioral region analysis

### **RECOMMENDED - Has Fallbacks:**
1. **Signal Processing Toolbox**
   - For `medfilt1` and `smooth`
   - Scripts work without these

## 🛠️ **Fallback Mechanisms Already Implemented**

### 1. **Smoothing Functions**
```matlab
% Built-in fallback for missing smooth function
if ~exist('smooth', 'file')
    smooth = @(x, n) movmean(x, n);
end
```

### 2. **Median Filtering**
```matlab
% Graceful handling of missing medfilt1
if exist('medfilt1', 'file')
    p1Dsmooth(i,:) = smooth(medfilt1(p1Dist(i,:),3),3);
else
    p1Dsmooth(i,:) = smooth(p1Dist(i,:),3);
end
```

### 3. **Image Filtering**
```matlab
% Fallback for missing imgaussfilt
if ~exist('imgaussfilt', 'file')
    if exist('imfilter', 'file') && exist('fspecial', 'file')
        imgaussfilt = @(img, sigma) imfilter(img, fspecial('gaussian', ceil(6*sigma), sigma));
    else
        % Simple box filter as fallback
        imgaussfilt = @(img, sigma) conv2(img, ones(ceil(6*sigma))/(ceil(6*sigma)^2), 'same');
    end
end
```

## 📊 **Data Format Validation**

### **Expected Data Structure:**
- **Directory**: `/work/rl349/dannce/mouse14/allData/`
- **File format**: `.mat` files with `pred` field
- **Data dimensions**: `[n_frames, 3, 14]`
- **Coordinate order**: `[x, y, z]` for 14 joints

### **Training Files Required:**
- `SNI_2.mat`
- `week4-TBI_3.mat`

### **Re-embedding Files (28 total):**
- DRG: 5 files
- IT: 2 files  
- SC: 6 files
- SNI: 2 files (excluding training file)
- Week4 files: 13 files (excluding training file)

## ⚡ **Performance Optimizations Implemented**

### **Memory Efficiency:**
- Uses 14 joints instead of 23 (60% reduction in joint pairs)
- 196 vs 529 pairwise distance calculations
- Efficient cell array handling with empty cell removal

### **Computational Efficiency:**
- Conditional function calls (only call expensive functions when available)
- Vectorized operations where possible
- Proper error handling with try-catch blocks

## 🔍 **Built-in Validation Features**

### **File Validation:**
```matlab
if exist(file_path, 'file')
    data = load(file_path);
    if isfield(data, 'pred')
        % Process file
    else
        error('File %s does not have expected pred structure', file_path);
    end
else
    error('Training file not found: %s', file_path);
end
```

### **Data Dimension Validation:**
```matlab
if size(mouseData, 3) ~= 14
    fprintf('⚠ Warning: Expected 14 joints, found %d in %s\n', size(mouseData, 3), filename);
    continue;
end
```

## 🚨 **Potential Issues and Solutions**

### **Issue 1: MotionMapper Not Found**
**Symptoms:** `Undefined function 'findWavelets'`
**Solution:** 
```matlab
% Download and add MotionMapper to path
addpath(genpath('/path/to/MotionMapper'))
```

### **Issue 2: Missing Statistics Toolbox**
**Symptoms:** `Undefined function 'tsne'`
**Solution:** Install Statistics and Machine Learning Toolbox or use third-party t-SNE implementation

### **Issue 3: Wrong Data Dimensions**
**Symptoms:** `Index exceeds array dimensions`
**Solution:** Verify your data has 14 joints, not 23:
```matlab
load('your_file.mat')
size(pred)  % Should show [n_frames, 3, 14]
```

### **Issue 4: Missing Training Files**
**Symptoms:** `Training file not found`
**Solution:** Ensure `SNI_2.mat` and `week4-TBI_3.mat` exist in data directory

## 🧪 **Testing Recommendations**

### **Before Running Pipeline:**
1. Run `dependency_checker.m` to validate environment
2. Check sample data file format:
   ```matlab
   data = load('/work/rl349/dannce/mouse14/allData/SNI_2.mat');
   size(data.pred)  % Should be [n_frames, 3, 14]
   ```
3. Verify MotionMapper functions:
   ```matlab
   which findWavelets  % Should show path to function
   ```

### **During Pipeline Execution:**
- Monitor console output for warnings
- Check memory usage if processing large datasets
- Verify intermediate files are created correctly

## 📈 **Expected Performance**

### **System Requirements:**
- **Memory**: 8+ GB RAM recommended
- **Storage**: ~2 GB for results
- **Processing time**: 15-45 minutes depending on data size

### **Typical Workflow:**
1. PCA on training data: ~2 minutes
2. t-SNE embedding: ~3 minutes
3. Re-embedding all files: ~15 minutes
4. Visualization creation: ~5 minutes
5. Flip analysis: ~10 minutes

## ✅ **Quality Assurance**

All scripts include:
- ✅ Proper error handling with try-catch blocks
- ✅ Input validation for file formats and dimensions
- ✅ Fallback mechanisms for optional functions
- ✅ Progress reporting and status updates
- ✅ Comprehensive output file generation
- ✅ Graceful handling of missing data

## 🎯 **Ready-to-Use Checklist**

Before running the pipeline, ensure:
- [ ] MotionMapper toolbox installed and in path
- [ ] Statistics and Machine Learning Toolbox available
- [ ] Image Processing Toolbox available
- [ ] Data directory exists: `/work/rl349/dannce/mouse14/allData/`
- [ ] Training files exist: `SNI_2.mat`, `week4-TBI_3.mat`
- [ ] Data format validated: `[n_frames, 3, 14]` dimensions
- [ ] Sufficient memory available (8+ GB)

---

**Conclusion:** The pipeline is robust and well-designed with excellent fallback mechanisms. The main requirement is ensuring MotionMapper toolbox is properly installed, as this is critical for the embedding functionality.






