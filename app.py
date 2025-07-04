import streamlit as st
import numpy as np
import os
from utils.predict import load_model, make_prediction
from utils.rag import initialize_rag, get_rag_response

# Page configuration
st.set_page_config(
    page_title="Breast Health Chat Assistant",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'welcome'
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

FEATURE_QUESTIONS = {
    'mean_radius': {
        'question': "Do you know the average size of the cells mentioned in your report? (usually between 6-28)",
        'help': "This refers to the mean radius of cell nuclei, typically measured in micrometers.",
        'range': (6.0, 28.0),
        'default': 14.0
    },
    'mean_texture': {
        'question': "Was there mention of how smooth or rough the cell surfaces looked? (scale 9-40)",
        'help': "Texture describes the variation in gray-scale values - smoother cells have lower values.",
        'range': (9.0, 40.0),
        'default': 19.0
    },
    'mean_perimeter': {
        'question': "Do you recall the perimeter measurement of the cells? (usually 40-190)",
        'help': "This is the distance around the edge of the cell nucleus.",
        'range': (40.0, 190.0),
        'default': 90.0
    },
    'mean_area': {
        'question': "Was there an area measurement mentioned? (typically 140-2500)",
        'help': "This measures the total area inside the cell nucleus boundary.",
        'range': (140.0, 2500.0),
        'default': 650.0
    },
    'mean_smoothness': {
        'question': "Did the report mention anything about cell smoothness? (range 0.05-0.16)",
        'help': "Smoothness measures how much the cell boundary varies from a perfect circle.",
        'range': (0.05, 0.16),
        'default': 0.10
    },
    'mean_compactness': {
        'question': "Was compactness discussed in your results? (range 0.02-0.35)",
        'help': "This measures how compact or dense the cell nucleus appears.",
        'range': (0.02, 0.35),
        'default': 0.10
    },
    'mean_concavity': {
        'question': "Did they mention anything about indentations in the cell shape? (range 0-0.43)",
        'help': "Concavity measures the severity of concave portions of the cell boundary.",
        'range': (0.0, 0.43),
        'default': 0.09
    },
    'mean_concave_points': {
        'question': "Were there notes about the number of indented areas? (range 0-0.20)",
        'help': "This counts the number of concave portions of the cell boundary.",
        'range': (0.0, 0.20),
        'default': 0.05
    },
    'mean_symmetry': {
        'question': "Was cell symmetry mentioned in your report? (range 0.11-0.30)",
        'help': "Symmetry measures how similar the cell looks when divided in half.",
        'range': (0.11, 0.30),
        'default': 0.18
    },
    'mean_fractal_dimension': {
        'question': "Did they discuss the complexity of the cell edges? (range 0.05-0.10)",
        'help': "Fractal dimension measures the complexity of the cell boundary pattern.",
        'range': (0.05, 0.10),
        'default': 0.06
    }
}

def display_welcome():
    st.markdown("""
    # 🎗️ Breast Health Chat Assistant
    
    ### Welcome! I'm here to help you understand your breast health information.
    
    **What I can do:**
    - Help you understand medical terms and measurements
    - Provide general information about breast health
    - Answer questions about symptoms and concerns
    - Offer supportive guidance (but not medical diagnosis)
    
    **Important Disclaimer:**
    This tool provides educational information only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.
    
    ---
    
    **How would you like to start?**
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 I have test results to discuss", use_container_width=True):
            st.session_state.current_step = 'collect_data'
            st.rerun()
    
    with col2:
        if st.button("❓ I have general questions", use_container_width=True):
            st.session_state.current_step = 'chat_mode'
            st.rerun()
    
    with col3:
        if st.button("📚 I want to learn more", use_container_width=True):
            st.session_state.current_step = 'education_mode'
            st.rerun()

def display_data_collection():
    """Display data collection interface"""
    st.markdown("## 📝 Let's Review Your Information Together")
    st.markdown("Don't worry if you don't have all the details - we can work with what you know!")
    
    progress = len(st.session_state.user_inputs) / len(FEATURE_QUESTIONS)
    st.progress(progress)
    
    # Display questions
    for feature, config in FEATURE_QUESTIONS.items():
        st.markdown(f"### {config['question']}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Input field
            current_value = st.session_state.user_inputs.get(feature, config['default'])
            
            value = st.number_input(
                f"Value for {feature.replace('_', ' ').title()}",
                min_value=float(config['range'][0]),  # Cast to float
                max_value=float(config['range'][1]),  # Cast to float
                value=float(current_value),           # Already float
                step=0.01,                            # Float step
                key=f"input_{feature}",
                label_visibility="collapsed"
            )
            
            st.session_state.user_inputs[feature] = value
        
        with col2:
            if st.button(f"ℹ️ Help", key=f"help_{feature}"):
                st.info(config['help'])
        
        # Skip option
        if st.checkbox(f"Skip this question", key=f"skip_{feature}"):
            st.session_state.user_inputs[feature] = config['default']
            st.info("Using typical average value")
        
        st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.user_inputs = {}
            st.rerun()
    
    with col2:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.session_state.current_step = 'welcome'
            st.rerun()
    
    with col3:
        if st.button("📊 Get Analysis", use_container_width=True, type="primary"):
            if len(st.session_state.user_inputs) >= 5:  # Minimum required
                st.session_state.current_step = 'show_results'
                st.rerun()
            else:
                st.warning("Please provide at least 5 measurements to get an analysis.")

def display_results():
    """Display prediction results"""
    st.markdown("## 📊 Your Analysis Results")
    
    try:
        # Load model and scaler - FIXED: properly unpack the tuple
        model, scaler = load_model()
        if model is None:
            st.error("Unable to load the analysis model. Please contact support.")
            return
        
        # Prepare input array
        input_array = np.array([
            st.session_state.user_inputs.get(feature, config['default']) 
            for feature, config in FEATURE_QUESTIONS.items()
        ]).reshape(1, -1)
        
        # Make prediction - FIXED: pass both model and scaler
        prediction, probability = make_prediction(model, input_array, scaler)
        
        # Display results with supportive messaging
        st.markdown("### 🎯 Analysis Complete")
        
        if prediction == 0:  # Benign
            st.success("✅ **Good News!** Based on the information provided, the indicators suggest everything looks normal.")
            st.markdown("""
            **What this means:**
            - The measurements you provided fall within typical ranges for healthy tissue
            - This is encouraging and suggests no immediate concerns
            - You're taking great care of your health by staying informed!
            
            **Confidence Level:** {:.1f}%
            """.format(probability * 100))
            
        else:  # Malignant
            st.warning("⚠️ **Important:** Some indicators suggest you should consult with a healthcare provider soon.")
            st.markdown("""
            **What this means:**
            - Some measurements fall outside typical ranges
            - This doesn't mean anything definitive - only a doctor can provide proper diagnosis
            - Many factors can cause these readings, and most are treatable
            - You're being proactive by seeking information, which is wonderful!
            
            **Confidence Level:** {:.1f}%
            """.format(probability * 100))
        
        # Always include reassuring information
        st.info("""
        **Remember:**
        - This analysis is based on limited information and should not replace professional medical advice
        - Early detection and regular check-ups are the best tools for maintaining breast health
        - You're taking positive steps by staying informed about your health
        """)
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💬 Ask Questions", use_container_width=True):
                st.session_state.current_step = 'chat_mode'
                st.rerun()
        
        with col2:
            if st.button("🔄 New Analysis", use_container_width=True):
                st.session_state.user_inputs = {}
                st.session_state.current_step = 'collect_data'
                st.rerun()
                
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.info("Please try again or contact support if the problem persists.")

def display_chat_mode():
    """Display chat interface with RAG"""
    st.markdown("## 💬 Ask Me Anything About Breast Health")
    
    # Initialize RAG system if not already done
    if st.session_state.rag_system is None:
        with st.spinner("Setting up knowledge base..."):
            st.session_state.rag_system = initialize_rag()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about breast health..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.rag_system:
                    response = get_rag_response(st.session_state.rag_system, prompt)
                else:
                    response = "I'm sorry, I'm having trouble accessing my knowledge base right now. Please try asking a simpler question or contact support."
                
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Suggested questions
    st.markdown("### 💡 Suggested Questions")
    suggestions = [
        "What does a lump mean?",
        "What if my mother had breast cancer?",
        "How accurate is a mammogram?",
        "What are the warning signs I should watch for?",
        "How often should I get checked?",
        "What lifestyle changes can help prevent breast cancer?"
    ]
    
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"suggestion_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": suggestion})
                st.rerun()

def display_education_mode():
    """Display educational content"""
    st.markdown("## 📚 Breast Health Education")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Understanding Results", "Prevention", "Risk Factors", "When to See a Doctor"])
    
    with tab1:
        st.markdown("""
        ### Understanding Your Test Results
        
        **Common Measurements Explained:**
        - **Cell Size (Radius):** How big the cells are on average
        - **Texture:** How smooth or rough the cell surfaces appear
        - **Perimeter:** The distance around the edge of cells
        - **Area:** The total space inside the cell boundary
        - **Smoothness:** How much the cell boundary varies from a perfect circle
        
        **What These Numbers Mean:**
        - Higher values don't automatically mean problems
        - Doctors look at patterns, not just individual numbers
        - Context from your medical history is crucial
        """)
    
    with tab2:
        st.markdown("""
        ### Prevention and Healthy Habits
        
        **Lifestyle Factors:**
        - Regular exercise (at least 150 minutes per week)
        - Maintain a healthy weight
        - Limit alcohol consumption
        - Don't smoke
        - Eat a diet rich in fruits and vegetables
        
        **Regular Screening:**
        - Monthly self-examinations
        - Annual clinical exams
        - Mammograms as recommended by your doctor
        """)
    
    with tab3:
        st.markdown("""
        ### Understanding Risk Factors
        
        **Factors You Can't Control:**
        - Age (risk increases with age)
        - Gender (women are at higher risk)
        - Family history
        - Genetic mutations (BRCA1, BRCA2)
        
        **Factors You Can Influence:**
        - Weight management
        - Physical activity
        - Alcohol consumption
        - Hormone therapy decisions
        """)
    
    with tab4:
        st.markdown("""
        ### When to Contact Your Doctor
        
        **See a Healthcare Provider If You Notice:**
        - A new lump or thickening
        - Changes in breast size or shape
        - Skin changes (dimpling, puckering)
        - Nipple discharge
        - Persistent breast pain
        
        **Remember:**
        - Most breast changes are not cancer
        - Early detection leads to better outcomes
        - You know your body best - trust your instincts
        """)

def main():
    """Main application logic"""
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎗️ Navigation")
        
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_step = 'welcome'
            st.rerun()
        
        if st.button("📝 Analysis", use_container_width=True):
            st.session_state.current_step = 'collect_data'
            st.rerun()
        
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.current_step = 'chat_mode'
            st.rerun()
        
        if st.button("📚 Learn", use_container_width=True):
            st.session_state.current_step = 'education_mode'
            st.rerun()
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        
        st.markdown("---")
        st.markdown("""
        ### ⚠️ Important Notice
        This tool provides educational information only. 
        Always consult healthcare professionals for medical advice.
        """)
    
    # Main content based on current step
    if st.session_state.current_step == 'welcome':
        display_welcome()
    elif st.session_state.current_step == 'collect_data':
        display_data_collection()
    elif st.session_state.current_step == 'show_results':
        display_results()
    elif st.session_state.current_step == 'chat_mode':
        display_chat_mode()
    elif st.session_state.current_step == 'education_mode':
        display_education_mode()

if __name__ == "__main__":
    main()