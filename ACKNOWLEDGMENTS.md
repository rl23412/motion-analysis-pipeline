# Acknowledgments

## 🤝 **Project Attribution**

This **Motion Analysis Pipeline** is built upon the excellent foundational work of several open-source projects and research contributions.

## 🔗 **Primary Attribution: SocialMapper**

### **Core Framework**
- **Project**: [SocialMapper](https://github.com/uklibaite/SocialMapper)
- **Description**: Combat-based motion mapper for behavioral analysis
- **Contribution**: Provided the complete MotionMapper utilities, embedding algorithms, and behavioral analysis framework that forms the backbone of this pipeline

### **Inherited Components**
- **Behavioral Embedding**: t-SNE, PCA, and wavelet-based feature extraction
- **Template Matching**: `findTemplatesFromData.m` and related algorithms
- **Re-embedding**: `findTDistributedProjections_fmin.m` for mapping new data
- **Watershed Segmentation**: `findWatershedRegions_v2.m` for behavioral region discovery
- **Utility Functions**: Complete set of helper functions and mathematical operations

## 🧬 **Scientific Methodology Attribution**

### **MotionMapper Framework**
**Original Paper**: 
> Berman, G.J., Choi, D.M., Bialek, W., Shaevitz, J.W. (2014). Mapping the stereotyped behaviour of freely moving fruit flies. *Journal of The Royal Society Interface*, 11(99).

**Contribution**: Foundational behavioral embedding methodology using wavelets and t-SNE.

### **ComBat Batch Correction**
**Original Paper**: 
> Johnson, W.E., Li, C., Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118-127.

**Contribution**: Empirical Bayes batch correction method for removing technical variation.

### **t-SNE Dimensionality Reduction**
**Original Paper**: 
> van der Maaten, L., Hinton, G. (2008). Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9(Nov), 2579-2607.

**Contribution**: Non-linear dimensionality reduction preserving local structure.

## 🔧 **Technical Dependencies**

### **MATLAB Toolboxes**
- **Statistics and Machine Learning Toolbox**: t-SNE implementation and statistical functions
- **Image Processing Toolbox**: Watershed segmentation and image processing
- **Signal Processing Toolbox**: Signal filtering and processing utilities

### **External Libraries**
- **SocialMapper Utilities**: Complete MotionMapper framework
- **ComBat**: Batch correction implementation
- **MEX Utilities**: Compiled functions for performance optimization

## 🎯 **Our Specialization**

While built on these excellent foundations, this pipeline adds:

### **Domain-Specific Adaptations**
- **Neuroscience Focus**: Specialized for pain research and behavioral phenotyping
- **ComBat Integration**: Advanced batch correction for multi-experiment datasets  
- **Professional Structure**: Software engineering best practices and modularity
- **Comprehensive Testing**: Validation framework and error handling
- **Enhanced Documentation**: Scientific context and usage guidance

### **Technical Improvements**
- **Modular Design**: Clean separation of training and analysis phases
- **Configuration System**: Easy parameter modification and experiment management
- **Robust Error Handling**: Comprehensive validation and fallback mechanisms
- **Visualization Suite**: Publication-ready figures and analysis outputs

## 🙏 **Gratitude Statement**

**We express our deepest gratitude to:**

1. **The SocialMapper Development Team** for creating an exceptional behavioral analysis framework that enabled this specialized application
2. **The MotionMapper Community** for developing the foundational embedding methodology
3. **The ComBat Authors** for providing robust batch correction methods
4. **The MATLAB Community** for excellent toolbox implementations
5. **The Open Science Community** for promoting reproducible research practices

## 📖 **How to Cite**

If you use this pipeline in your research, please cite both this work and the foundational methods:

### **This Pipeline**
```bibtex
@software{motion_analysis_pipeline_2025,
  title={Motion Analysis Pipeline with ComBat Batch Correction},
  author={[Your Name]},
  year={2025},
  url={https://github.com/rl23412/motion-analysis-pipeline}
}
```

### **SocialMapper Foundation**
```bibtex
@software{socialmapper_2023,
  title={SocialMapper: Combat-based motion mapper},
  url={https://github.com/uklibaite/SocialMapper},
  year={2023}
}
```

### **MotionMapper Methodology**
```bibtex
@article{berman2014mapping,
  title={Mapping the stereotyped behaviour of freely moving fruit flies},
  author={Berman, Gordon J and Choi, Daniel M and Bialek, William and Shaevitz, Joshua W},
  journal={Journal of The Royal Society Interface},
  volume={11},
  number={99},
  year={2014}
}
```

### **ComBat Method**
```bibtex
@article{johnson2007adjusting,
  title={Adjusting batch effects in microarray expression data using empirical Bayes methods},
  author={Johnson, W Evan and Li, Cheng and Rabinovic, Ariel},
  journal={Biostatistics},
  volume={8},
  number={1},
  pages={118--127},
  year={2007}
}
```

## 🌟 **Community Spirit**

This project exemplifies the power of open science and collaborative research. By building upon excellent foundational work while adding specialized functionality, we aim to:

- **Advance Scientific Research**: Enable new discoveries in neuroscience and behavioral analysis
- **Promote Reproducibility**: Provide well-documented, tested, and validated tools
- **Foster Collaboration**: Support the broader behavioral analysis community
- **Maintain Attribution**: Properly credit all foundational contributions

## 🚀 **Future Collaboration**

We remain committed to:
- **Contributing Back**: Sharing improvements that benefit the broader community
- **Maintaining Attribution**: Ensuring proper credit in all derivative works
- **Supporting Users**: Providing documentation and assistance for researchers
- **Open Development**: Transparent development and peer review

**This pipeline stands as a testament to the collaborative spirit of computational biology and the power of building upon excellent foundational work.**

---

*For questions about this pipeline, please use our repository's issue tracker. For questions about the foundational SocialMapper framework, please refer to their repository.*