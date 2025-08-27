# Relationship to SocialMapper Project

## 🔗 **Project Lineage**

This **Motion Analysis Pipeline** is **directly adapted from [SocialMapper](https://github.com/uklibaite/SocialMapper)**, a combat-based motion mapping framework. We acknowledge and appreciate the foundational work that made this specialized application possible.

### **Original Project**
- **Repository**: [https://github.com/uklibaite/SocialMapper](https://github.com/uklibaite/SocialMapper)
- **Description**: Combat-based motion mapper built on MotionMapper framework
- **Core Innovation**: Behavioral embedding and analysis for combat behaviors

### **Our Adaptation**
- **Repository**: [https://github.com/rl23412/motion-analysis-pipeline](https://github.com/rl23412/motion-analysis-pipeline)
- **Specialization**: Spontaneous pain behavioral analysis in neuroscience research
- **Domain**: Rodent pain models and behavioral phenotyping

---

## 🧬 **What We Inherited**

### **Core Methodologies** (from SocialMapper/MotionMapper)
1. **Behavioral Embedding Framework**
   - t-SNE dimensionality reduction
   - Wavelet-based feature extraction
   - PCA preprocessing pipelines
   - Watershed segmentation for behavioral regions

2. **Data Processing Architecture**
   - Batch processing workflows
   - Template matching algorithms
   - Re-embedding techniques for new data
   - Statistical analysis frameworks

3. **Code Structure and Organization**
   - Modular MATLAB pipeline design
   - Configuration management patterns  
   - Error handling approaches
   - Visualization and reporting tools

### **Key Functions Adapted**
```matlab
% Core functions inherited from MotionMapper ecosystem:
findWavelets.m                    % Wavelet decomposition
findTemplatesFromData.m           % Template extraction  
findTDistributedProjections_fmin.m % Re-embedding
findWatershedRegions_v2.m         % Region segmentation
setRunParameters.m                % Configuration
```

---

## 🎯 **What We Specialized**

### **Domain-Specific Adaptations**
1. **Pain Behavioral Analysis**
   - Pain-specific behavioral metrics
   - Spontaneous behavior characterization
   - Clinical relevance scoring

2. **Left-Right Data Augmentation**
   - Mirror image generation for pose data
   - Joint mapping for symmetrical behaviors  
   - Enhanced statistical power for small samples

3. **Neuroscience Integration**
   - DANNCE pose estimation compatibility
   - Experimental group management
   - Research workflow optimization

4. **Enhanced Documentation**
   - Scientific context and methodology
   - Neuroscience-specific usage examples
   - Pain research applications and validation

### **New Features Added**
```matlab
% Specialized functions for pain analysis:
mouse_embedding.m              % Enhanced embedding with L-R flipping
analyze_maps_and_counts.m      % Pain-specific analysis workflows
group_config.m                 % Experimental group management
```

---

## 📊 **Technical Comparison**

| Aspect | SocialMapper | This Pipeline |
|--------|-------------|--------------|
| **Domain** | Combat behavior | Spontaneous pain |
| **Species** | General | Rodent models |
| **Data Aug** | Standard | Left-right flipping |
| **Groups** | Combat scenarios | Experimental conditions |
| **Output** | Combat analysis | Pain metrics & CSV |
| **Documentation** | Technical | Scientific + clinical |

---

## 🤝 **Attribution and Respect**

### **How We Honor the Original Work**
1. **Clear Attribution**: Links and citations throughout documentation
2. **Preserved Credit**: Maintain references to original authors
3. **Scientific Integrity**: Acknowledge methodological foundations
4. **Community Spirit**: Contribute back improvements where applicable

### **Collaborative Approach**
- **Independent Development**: Maintains specialized focus on pain research
- **Shared Foundations**: Benefits from continued MotionMapper improvements
- **Cross-Pollination**: Techniques may benefit broader motion analysis community
- **Open Science**: Both projects support reproducible research

---

## 📚 **Scientific Context**

### **Why This Adaptation Was Needed**
1. **Domain Expertise**: Pain research requires specialized metrics and workflows
2. **Clinical Relevance**: Different behavioral patterns and significance
3. **Experimental Design**: Neuroscience-specific group comparisons and statistics
4. **Data Characteristics**: Different pose data sources and preprocessing needs

### **Value Added**
- **Specialized Algorithms**: Pain-specific behavioral characterization
- **Enhanced Workflows**: Streamlined for neuroscience research labs
- **Validation Framework**: Pain research-specific validation and testing
- **Documentation**: Domain-specific guides and troubleshooting

---

## 🔬 **Research Impact**

### **SocialMapper's Contribution to Our Work**
- **Methodological Foundation**: Solid behavioral embedding framework
- **Technical Excellence**: Robust, tested algorithms and implementations  
- **Code Quality**: Professional MATLAB pipeline architecture
- **Community Standards**: Open science and reproducibility practices

### **Our Contribution to the Field**
- **Domain Specialization**: Advancing pain research methodologies
- **Data Augmentation**: Novel left-right flipping for behavioral data
- **Clinical Translation**: Streamlined workflows for research applications
- **Open Science**: Fully documented, tested, and shareable implementation

---

## 🚀 **Future Collaboration**

We remain open to:
- **Technical Improvements**: Sharing optimizations beneficial to both projects
- **Methodological Advances**: Contributing general motion analysis improvements
- **Community Building**: Supporting the broader behavioral analysis ecosystem
- **Scientific Exchange**: Collaborative research opportunities

---

## 🙏 **Gratitude Statement**

**We are deeply grateful to the SocialMapper development team** for creating an excellent foundation that enabled our specialized pain research application. Their commitment to open science and high-quality implementations made this work possible.

**This pipeline stands as a testament to the power of open-source collaboration in advancing scientific research.**

---

*For technical questions about the original SocialMapper framework, please refer to their repository. For questions about this pain analysis specialization, please use our repository's issue tracker.*

**Both projects contribute to the advancement of computational behavioral analysis and open science principles.**