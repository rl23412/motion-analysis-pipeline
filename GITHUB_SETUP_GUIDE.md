# GitHub Setup Guide: Spontaneous Pain Analysis Pipeline

This guide will help you set up the pipeline as a GitHub repository and create a relationship with the SocialMapper project.

## 🎯 **Setup Options**

### Option 1: Create as Independent Repository (Recommended)
Since your pipeline is significantly different from SocialMapper, create it as a standalone project with attribution.

### Option 2: Fork Relationship  
If you want to maintain connection to SocialMapper, we'll set up proper attribution.

---

## 📋 **Step-by-Step Instructions**

### **Step 1: Create GitHub Repository**

1. **Go to GitHub.com** and log into your account
2. **Click "New Repository"** (+ button in top right)
3. **Repository Settings**:
   - **Name**: `spontaneous-pain-analysis-pipeline`
   - **Description**: `MATLAB pipeline for spontaneous pain behavioral analysis using DANNCE pose estimation and left-right flipping augmentation`
   - **Visibility**: Public or Private (your choice)
   - **DO NOT** initialize with README (we already have one)

### **Step 2: Connect Local Repository to GitHub**

Open Command Prompt/Terminal in your project directory and run:

```bash
cd "C:\Users\Runda\Desktop\activities\Ji Lab work\2025summer\archives\DANNCE\spontaneous-pain-pipeline"

# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/spontaneous-pain-analysis-pipeline.git

# Push to GitHub  
git branch -M main
git push -u origin main
```

### **Step 3: Add Attribution to SocialMapper** 

I'll add proper attribution to acknowledge the SocialMapper project:

```bash
# Add attribution in README and create ACKNOWLEDGMENTS file
```

---

## 🔗 **SocialMapper Attribution**

Since your pipeline builds upon concepts from SocialMapper but is substantially different:

1. **Not a traditional fork** - Your pipeline is domain-specific (pain analysis)
2. **Proper attribution** - We'll acknowledge SocialMapper in documentation
3. **Independent project** - Maintains your own development path

---

## ⚡ **Quick Commands for You**

**Replace `YOUR_USERNAME` with your actual GitHub username:**

```bash
# Navigate to project
cd "C:\Users\Runda\Desktop\activities\Ji Lab work\2025summer\archives\DANNCE\spontaneous-pain-pipeline"

# Add remote (REPLACE YOUR_USERNAME!)
git remote add origin https://github.com/YOUR_USERNAME/spontaneous-pain-analysis-pipeline.git

# Push to GitHub
git push -u origin main
```

---

## 🚀 **What Happens Next**

After pushing, your GitHub repository will have:
- ✅ Complete project structure
- ✅ All source code and documentation  
- ✅ Git history with professional commits
- ✅ README with installation/usage instructions
- ✅ Test suite and validation tools
- ✅ Professional .gitignore and LICENSE

---

## 🤝 **If You Want Help**

**I can help by:**
1. ✅ Creating attribution files
2. ✅ Writing GitHub-specific documentation
3. ✅ Setting up GitHub Actions (CI/CD)
4. ✅ Creating issue templates
5. ✅ Adding contribution guidelines

**You'll need to do:**
1. Create the GitHub repository (web interface)
2. Run the git commands (I can't access your account)
3. Provide your GitHub username for the commands

---

## 🔐 **Security Note**

**I cannot and will not:**
- Access your GitHub account
- Handle passwords or tokens
- Push directly to GitHub

This ensures your account security while still helping you set up the project professionally.

---

**Ready to proceed? Just let me know your GitHub username and I'll provide the exact commands!**