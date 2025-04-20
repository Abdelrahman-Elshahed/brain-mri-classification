import streamlit as st
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Configure the page
st.set_page_config(
    page_title="Brain MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FastAPI endpoint URL (change if deployed elsewhere)
API_URL = "http://localhost:8000"

def main():
    # Add header with logo and title
    st.title("🧠 Brain MRI Classification System")
    st.markdown("""
    This application classifies brain MRI scans using two deep learning models:
    - **VGG16 Model** (transfer learning with fine-tuning)
    - **ResNet18 Model** (transfer learning with fine-tuning)
    
    The models detect four classes of brain conditions:
    - Glioma (tumor)
    - Meningioma (tumor)
    - No tumor (normal)
    - Pituitary tumor
    """)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Model Comparison", "About"])
    
    # Check API connection
    try:
        health_response = requests.get(f"{API_URL}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.sidebar.success("✅ Connected to API")
            st.sidebar.info(f"Backend running on: {health_data.get('device', 'N/A')}")
        else:
            st.sidebar.error("❌ API is not responding properly")
    except:
        st.sidebar.error("❌ Cannot connect to API")
        st.sidebar.info(f"Make sure FastAPI is running at {API_URL}")
    
    if page == "Home":
        show_home_page()
    elif page == "Model Comparison":
        show_model_comparison()
    else:
        show_about_page()

def show_home_page():
    # File uploader
    uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Uploaded MRI Scan", width=300)
        
        with col2:
            # Add a prediction button
            predict_button = st.button("Classify MRI Scan")
            
            if predict_button:
                # Show spinner during prediction
                with st.spinner("Analyzing the MRI scan..."):
                    # Reset file pointer and get file bytes
                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.getvalue()
                    
                    # Make parallel calls to both models
                    try:
                        # Call VGG16 model
                        vgg_response = requests.post(
                            f"{API_URL}/predict/vgg",
                            files={"file": ("image.jpg", file_bytes, "image/jpeg")}
                        )
                        
                        # Call ResNet18 model
                        resnet_response = requests.post(
                            f"{API_URL}/predict/resnet",
                            files={"file": ("image.jpg", file_bytes, "image/jpeg")}
                        )
                        
                        if vgg_response.status_code == 200 and resnet_response.status_code == 200:
                            vgg_results = vgg_response.json()
                            resnet_results = resnet_response.json()
                            
                            # Compare results and display
                            combined_results = {
                                "vgg16": vgg_results,
                                "resnet18": resnet_results,
                                "agreement": vgg_results["class"] == resnet_results["class"]
                            }
                            
                            display_results(combined_results, image)
                        else:
                            if vgg_response.status_code != 200:
                                st.error(f"Error from VGG16 API: {vgg_response.status_code}")
                                st.error(vgg_response.text)
                            if resnet_response.status_code != 200:
                                st.error(f"Error from ResNet18 API: {resnet_response.status_code}")
                                st.error(resnet_response.text)
                    except Exception as e:
                        st.error(f"Failed to connect to API: {str(e)}")
                        st.info("Make sure the FastAPI server is running at " + API_URL)

def display_results(results, image):
    # Extract results
    vgg_results = results["vgg16"]
    resnet_results = results["resnet18"]
    agreement = results["agreement"]
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Results Summary", "Detailed Analysis"])
    
    with tab1:
        # Show prediction results in a clean layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("VGG16 Model")
            st.markdown(f"**Prediction:** {vgg_results['class']}")
            st.markdown(f"**Confidence:** {vgg_results['confidence']:.2%}")
            st.markdown(f"**Inference time:** {vgg_results.get('inference_time', 'N/A')} seconds")
            
            # Display confidence bar
            st.progress(vgg_results['confidence'])
        
        with col2:
            st.subheader("ResNet18 Model")
            st.markdown(f"**Prediction:** {resnet_results['class']}")
            st.markdown(f"**Confidence:** {resnet_results['confidence']:.2%}")
            st.markdown(f"**Inference time:** {resnet_results.get('inference_time', 'N/A')} seconds")
            
            # Display confidence bar
            st.progress(resnet_results['confidence'])
        
        # Show agreement message
        if agreement:
            st.success("✅ Both models agree on the classification!")
        else:
            st.warning("⚠️ The models disagree on the classification. Consider consulting a specialist.")
            
        # Add explanation of the predicted condition
        st.subheader("About the Diagnosis")
        condition = vgg_results['class'] if agreement else "Uncertain"
        
        conditions = {
            "glioma": """
                **Glioma** is a type of tumor that originates in the glial cells of the brain or spine. 
                Glial cells support and nourish neurons. Gliomas are often aggressive and can be difficult to treat.
            """,
            "meningioma": """
                **Meningioma** is usually a benign (non-cancerous) tumor that arises from the meninges - 
                the membranes that surround the brain and spinal cord. Most meningiomas grow slowly.
            """,
            "notumor": """
                **No tumor detected**. The scan appears to show normal brain tissue without evidence of tumors.
            """,
            "pituitary": """
                **Pituitary tumor** affects the pituitary gland at the base of the brain. 
                These tumors can affect hormone production and may cause various symptoms related to hormone imbalances.
            """,
            "Uncertain": """
                **Uncertain diagnosis**. The models have provided different classifications. 
                Please consult with a medical professional for proper diagnosis.
            """
        }
        
        st.markdown(conditions.get(condition.lower(), conditions["Uncertain"]))
    
    with tab2:
        # Show model confidence comparison
        st.subheader("Model Confidence Comparison")
        
        # Create dataframe for bar chart
        classes = ["glioma", "meningioma", "notumor", "pituitary"]
        
        # Since we don't have full probability distributions from the API,
        # we'll create a simple bar chart showing the confidence of each model's prediction
        
        fig = go.Figure()
        
        model_data = [
            {"Model": "VGG16", "Class": vgg_results['class'], "Confidence": vgg_results['confidence']},
            {"Model": "ResNet18", "Class": resnet_results['class'], "Confidence": resnet_results['confidence']}
        ]
        
        df = pd.DataFrame(model_data)
        
        fig = px.bar(
            df, 
            x="Model", 
            y="Confidence", 
            color="Class",
            title="Model Confidence for Predicted Classes",
            text_auto='.2%'
        )
        
        fig.update_layout(yaxis_range=[0, 1])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show inference time comparison
        if 'inference_time' in vgg_results and 'inference_time' in resnet_results:
            st.subheader("Inference Time Comparison")
            
            time_data = [
                {"Model": "VGG16", "Time (seconds)": vgg_results['inference_time']},
                {"Model": "ResNet18", "Time (seconds)": resnet_results['inference_time']}
            ]
            
            time_df = pd.DataFrame(time_data)
            
            time_fig = px.bar(
                time_df,
                x="Model",
                y="Time (seconds)",
                title="Model Inference Time",
                text_auto='.3f'
            )
            
            st.plotly_chart(time_fig, use_container_width=True)

def show_model_comparison():
    st.header("Model Comparison")
    
    # Add a performance metrics section with actual results from classification reports
    st.subheader("Performance Metrics")
    
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision (macro avg)', 'Recall (macro avg)', 'F1-Score (macro avg)'],
        'VGG16': ['97%', '98%', '97%', '97%'],
        'ResNet18': ['98%', '98%', '98%', '98%']
    })
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Add class-specific metrics
    st.subheader("Class-specific Performance")
    
    vgg_class_df = pd.DataFrame({
        'Class': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
        'Precision': ['100%', '97%', '96%', '98%'],
        'Recall': ['94%', '95%', '100%', '99%'],
        'F1-Score': ['97%', '96%', '98%', '99%']
    })
    
    resnet_class_df = pd.DataFrame({
        'Class': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
        'Precision': ['100%', '92%', '100%', '100%'],
        'Recall': ['92%', '100%', '100%', '99%'],
        'F1-Score': ['96%', '96%', '100%', '100%']
    })
    
    tab1, tab2 = st.tabs(["VGG16 Performance", "ResNet18 Performance"])
    
    with tab1:
        st.dataframe(vgg_class_df, use_container_width=True)
        
    with tab2:
        st.dataframe(resnet_class_df, use_container_width=True)
    
    # Create a bar chart comparing accuracy
    fig = px.bar(
        x=['VGG16', 'ResNet18'],
        y=[0.97, 0.98],
        labels={'x': 'Model', 'y': 'Accuracy'},
        title='Model Accuracy Comparison',
        text_auto='.0%',
        color=['VGG16', 'ResNet18'],
        color_discrete_map={'VGG16': '#4285f4', 'ResNet18': '#34a853'}
    )
    fig.update_layout(yaxis_range=[0.9, 1.0])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Architecture Comparison
    
    | Feature | VGG16 | ResNet18 |
    |---------|-------|----------|
    | **Base Architecture** | Very Deep Network with 16 layers | Residual Network with 18 layers |
    | **Special Features** | Simple architecture with successive conv layers | Skip connections to solve vanishing gradient |
    | **Parameters** | ~138M (full model) | ~11.7M (full model) |
    | **Image Input Size** | 128x128 | 224x224 |
    | **Transfer Learning** | Uses ImageNet weights | Uses ImageNet weights |
    | **Fine-tuning** | Last block fine-tuned | Last layer fine-tuned |
    
    ### Performance Characteristics
    
    | Aspect | VGG16 | ResNet18 |
    |--------|-------|----------|
    | **Accuracy** | 97% | Slightly higher at 98% |
    | **Inference Speed** | Slower due to more parameters | Faster due to fewer parameters |
    | **Model Size** | Larger (~500MB) | Smaller (~45MB) |
    | **Resource Usage** | Higher memory usage | Lower memory usage |
    """)
    
    st.subheader("When to use which model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **VGG16 is better for:**
        - More consistent class-wise performance
        - Better recall for glioma detection (94%)
        - When interpretability is important (simpler architecture)
        - When computing resources aren't constrained
        """)
    
    with col2:
        st.markdown("""
        **ResNet18 is better for:**
        - Higher overall accuracy (98%)
        - Perfect precision and recall for no tumor detection (100%)
        - Faster inference times
        - Deployment on resource-constrained devices
        """)

def show_about_page():
    st.header("About")
    
    st.markdown("""
    ### Brain MRI Classification Project
    
    This application uses deep learning models to classify brain MRI scans into four categories:
    - **Glioma**: A type of tumor that starts in the glial cells of the brain
    - **Meningioma**: A tumor that forms on membranes covering the brain and spinal cord
    - **No tumor**: Normal brain tissue
    - **Pituitary**: A tumor that forms in the pituitary gland
    
    ### Model Performance
    
    Our models achieved excellent performance on the test dataset:
    
    - **VGG16**: 97% accuracy with strong performance across all classes
    - **ResNet18**: 98% accuracy with perfect performance on "no tumor" detection
    
    **Key Strengths:**
    - VGG16 excels at detecting glioma and meningioma tumors
    - ResNet18 achieves 100% precision and recall for no tumor detection
    - Both models achieve over 99% performance for pituitary tumor detection
    
    ### Technology Stack
    
    - **Frontend**: Streamlit
    - **Backend**: FastAPI
    - **Deep Learning Frameworks**: TensorFlow (VGG16) and PyTorch (ResNet18)
    - **Models**: VGG16 and ResNet18 with transfer learning
    
    ### Data Source
    
    The models were trained on the Brain Tumor MRI Dataset from Kaggle, containing T1-weighted contrast-enhanced images.
    
    """)

if __name__ == "__main__":
    main()