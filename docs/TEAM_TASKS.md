# Team Task Assignment Guide

This document provides a comprehensive guide for two new team members joining the project.

---

## 📋 Task Overview

| Role | Primary Task | Related Files/Directories |
|------|-------------|-------------------------|
| **Data Collection Lead** | Collect and preprocess balanced training data | `src/data/loader.py`, `data/raw/` |
| **Microcontroller Deployment Lead** | Deploy models to microcontroller (ESP32 or Raspberry Pi Pico) and measure performance | `esp32_tflite_project/`, `scripts/deploy_microcontroller.py` |

---

## 📊 Task 1: Data Collection Lead

### Objective
**Research and collect balanced training datasets for the project**

### Current Status
- Currently using datasets: Bot-IoT, CIC-IDS2017
- Issue: Potential class imbalance (attack samples may be underrepresented)
- Goal: Find balanced datasets suitable for IoT security intrusion detection

### Tasks

#### Step 1: Dataset Research
- [ ] **Research balanced IoT security datasets**
  - Recommended datasets to investigate:
    - **UNSW-NB15**: Network traffic data with relatively balanced classes
    - **NSL-KDD**: Improved version of KDD Cup 99 with more balanced class distribution
    - **CIC-IDS2017**: Already in use, but research more balanced subsets
    - **ToN-IoT**: Recent IoT attack dataset
    - **CIC-DDoS2019**: DDoS attack dataset
    - **Mirai Botnet Dataset**: IoT botnet traffic data
  - Dataset requirements:
    - CSV format (or easily convertible)
    - Network traffic/packet data
    - Attack/normal labels included
    - Class ratio within 1:3 (attack:normal) minimum preferred
    - Publicly available or accessible

- [ ] **Document dataset findings**
  - Create a research document with:
    - Dataset name and source
    - Dataset size and format
    - Class distribution information
    - Download links and access requirements
    - License and usage terms
    - Balance assessment (class ratios)

#### Step 2: Dataset Collection
- [ ] **Download selected datasets**
  ```bash
  # Data storage location
  data/raw/[dataset_name]/
  ```

- [ ] **Handle large datasets (GitHub size limits)**
  - **Important**: Large datasets (>100MB) should NOT be committed to Git
  - **Recommended approach**: Use Google Drive for large datasets
    1. **Upload to Google Drive**:
       - Upload dataset files to a shared Google Drive folder
       - Ensure proper sharing permissions (team members can access)
       - Create a direct download link or shareable link
    2. **Document in research document**:
       - Include Google Drive link in `docs/DATASET_RESEARCH.md`
       - Provide download instructions
       - Note file sizes and storage location

- [ ] **Verify dataset integrity**
  - Check file completeness
  - Verify data format matches expectations
  - Confirm labels are present and correct
  - Note file sizes in documentation

- [ ] **Basic balance check** (optional, if time permits)
  - Quick check: Count samples per class if dataset is small enough
  - Note: Detailed analysis will be done by the team later
  - Document any obvious imbalance issues found

#### Step 3: Documentation
- [ ] **Create dataset summary document**
  - File: `docs/DATASET_RESEARCH.md` (create new)
  - Contents:
    - List of researched datasets
    - Selected datasets with justification
    - Class distribution analysis
    - Download instructions:
      - **Google Drive links** for large datasets (>100MB)
      - Direct download links if available
      - Access instructions and permissions
    - Storage locations (local path: `data/raw/[dataset_name]/`)
    - File sizes and format information
    - Note if dataset is stored externally (Google Drive) and why

### Deliverables
- Research document with at least 5-7 potential datasets
- At least 3-4 balanced datasets downloaded and stored in `data/raw/` (or Google Drive for large files)
- Dataset summary document with findings

### Reference Materials
- Existing datasets: Bot-IoT, CIC-IDS2017 in `data/raw/`
- Balance check script: `scripts/check_data_balance.py`
- Dataset download guide: `scripts/download_dataset.sh`

### Checklist
- [ ] Research at least 5-7 potential balanced datasets
- [ ] Document findings and class distributions
- [ ] Download at least 3-4 selected datasets
- [ ] Store datasets in `data/raw/` directory (or Google Drive for large files)
- [ ] Verify dataset integrity and format
- [ ] Create dataset research summary document

---

## 🔌 Task 2: Microcontroller Deployment Lead

### Objective
**Deploy compressed TFLite models to microcontroller hardware and measure real-world performance**

### Current Status
- ✅ ESP32 project structure ready: `esp32_tflite_project/`
- ✅ TFLite → C array conversion script: `scripts/deploy_microcontroller.py`
- ✅ Local inference test script: `scripts/test_tflite_inference.py`
- ⏳ Actual hardware deployment and testing needed

### Tasks

#### Step 1: Hardware Platform Selection
- [ ] **Choose hardware platform** (select one):
  - **ESP32**: Existing project structure available
  - **Raspberry Pi Pico**: Alternative option, may require additional setup
  - Consider factors: memory constraints, development tools, project requirements

#### Step 2: Environment Setup
- [ ] **Set up development environment**
  - Install required tools (PlatformIO, SDK, etc.)
  - Prepare hardware board and necessary cables
  - Verify project directory and dependencies

#### Step 3: Model Preparation
- [ ] **Generate compressed TFLite model**
  - Run compression pipeline or use existing compressed model
  - Target model size: **≤ 100KB** (considering microcontroller memory constraints)
  - Verify model size and format

- [ ] **Convert to C array**
  - Use `scripts/deploy_microcontroller.py` to convert TFLite model
  - Verify generated files and model compatibility

#### Step 4: Hardware Deployment
- [ ] **Modify microcontroller code**
  - Update model references and configurations
  - Adjust memory buffers and tensor sizes
  - Implement inference logic

- [ ] **Build and upload**
  - Build project successfully
  - Upload to hardware
  - Verify deployment

#### Step 5: Performance Measurement
- [ ] **Measure inference latency**
  - Test with various input data
  - Record inference times
  - Compare with software inference

- [ ] **Analyze memory usage**
  - Measure memory footprint
  - Verify within hardware limits

- [ ] **Real-world performance evaluation**
  - Test normal vs attack traffic classification
  - Verify accuracy (if possible)
  - Document any performance issues

#### Step 6: Documentation
- [ ] **Update deployment guide**
  - Update `docs/MICROCONTROLLER_DEPLOYMENT.md`
  - Document hardware-specific findings

- [ ] **Create performance report**
  - File: `docs/MICROCONTROLLER_DEPLOYMENT_REPORT.md` (create new)
  - Include: model size, inference time, memory usage, accuracy comparison, issues and solutions

### Reference Materials
- ESP32 project: `esp32_tflite_project/`
- Deployment script: `scripts/deploy_microcontroller.py`
- Deployment guide: `docs/MICROCONTROLLER_DEPLOYMENT.md`
- PlatformIO documentation: https://platformio.org/

### Checklist
- [ ] Select hardware platform (ESP32 or Raspberry Pi Pico)
- [ ] Set up development environment
- [ ] Generate compressed TFLite model (≤ 100KB)
- [ ] Convert model to C array
- [ ] Deploy to hardware
- [ ] Measure inference latency
- [ ] Analyze memory usage
- [ ] Evaluate real-world performance
- [ ] Document findings

---

## 🤝 Collaboration Guide

### Workflow Order
1. **Data Collection Lead** prepares balanced data first
2. **Microcontroller Deployment Lead** can test with existing models first
3. After retraining with new data, Microcontroller Deployment Lead deploys updated model

### Communication
- Share work progress via issues or PRs
- Notify team immediately when issues arise
- Weekly progress reports

### Code Review
- Each lead's work should be reviewed by other team members
- Test code writing recommended
- Documentation updates required

---

## 📝 Checklist Summary

### Data Collection Lead
- [ ] Research balanced datasets
- [ ] Download and store datasets
- [ ] Document findings and create summary

### ESP Deployment Lead
- [ ] Environment setup (PlatformIO, ESP32)
- [ ] Prepare model and convert to C array
- [ ] Modify ESP32 code and build
- [ ] Hardware testing and performance measurement
- [ ] Documentation

---

**Created**: 2026-01-27  
**Last Updated**: Update this document as needed.
