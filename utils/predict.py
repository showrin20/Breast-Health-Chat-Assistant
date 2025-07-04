import joblib
import numpy as np
import os
from typing import Tuple, Optional, Dict
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path: str = os.path.join('model', 'svm_model.joblib'), 
               scaler_path: Optional[str] = os.path.join('model', 'scaler.joblib')) -> Tuple[Optional[object], Optional[object]]:
    """
    Load a pre-trained SVM model and optional scaler from files.

    Args:
        model_path: Path to the model file (default: 'model/svm_model.joblib')
        scaler_path: Path to the scaler file (default: 'model/scaler.joblib')

    Returns:
        Tuple containing the loaded model and scaler (or None if loading fails)
    """
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return None, None
        
        model = joblib.load(model_path)
        scaler = None
        if scaler_path and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        else:
            logger.warning(f"Scaler file not found at {scaler_path}. Predictions may be inaccurate if scaling is required.")
        
        logger.info(f"Model loaded successfully from {model_path}")
        return model, scaler
    
    except Exception as e:
        logger.error(f"Error loading model or scaler: {str(e)}")
        return None, None

def make_prediction(model: object, input_array: np.ndarray, scaler: Optional[object] = None) -> Tuple[int, float]:
    """
    Make a prediction using the SVM model and return the class and confidence score.

    Args:
        model: Loaded SVM model
        input_array: Input features as a NumPy array (shape: (1, 30))
        scaler: Optional scaler to transform input features

    Returns:
        Tuple of (predicted class, confidence score)
    """
    try:
        if model is None:
            logger.error("No valid model provided for prediction")
            return 0, 0.5

        # Ensure input has correct shape (1, 30)
        if input_array.shape[1] != 30:
            logger.error(f"Input array has {input_array.shape[1]} features, expected 30")
            return 0, 0.5

        # Apply scaling if scaler is provided
        if scaler is not None:
            input_array = scaler.transform(input_array)
        
        prediction = model.predict(input_array)[0]
        
        # Calculate confidence score
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_array)[0]
            confidence = float(max(probabilities))
        elif hasattr(model, 'decision_function'):
            decision_score = model.decision_function(input_array)[0]
            confidence = float(1 / (1 + np.exp(-decision_score)))  # Sigmoid conversion
        else:
            logger.warning("Model does not support probability or decision function. Defaulting confidence to 0.5")
            confidence = 0.5
        
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
        return int(prediction), confidence
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return 0, 0.5

def validate_input(input_dict: Dict[str, float]) -> bool:
    """
    Validate user input values for breast cancer prediction.

    Args:
        input_dict: Dictionary of feature values

    Returns:
        Boolean indicating if input is valid
    """
    required_features = [
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
        'mean_smoothness', 'mean_compactness', 'mean_concavity',
        'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension'
    ]
    
    # Check if all required features are present
    for feature in required_features:
        if feature not in input_dict:
            logger.error(f"Missing required feature: {feature}")
            return False
        
        # Check if value is numeric and within realistic ranges
        try:
            value = float(input_dict[feature])
            if value < 0:  # Most features should be non-negative
                logger.error(f"Invalid value for {feature}: {value}. Must be non-negative.")
                return False
            # Add specific range checks if known (example ranges for Wisconsin dataset)
            if feature == 'mean_radius' and not (6.0 <= value <= 30.0):
                logger.warning(f"Value for {feature} ({value}) is outside typical range (6.0-30.0).")
        except (ValueError, TypeError):
            logger.error(f"Invalid value for {feature}: {input_dict[feature]}. Must be numeric.")
            return False
    
    return True

def preprocess_input(input_dict: Dict[str, float]) -> np.ndarray:
    """
    Preprocess input dictionary into a NumPy array with all 30 features for prediction.
    This function takes 10 mean features and generates SE and worst features.

    Args:
        input_dict: Dictionary of mean feature values

    Returns:
        NumPy array with shape (1, 30) containing all features
    """
    mean_features = [
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
        'mean_smoothness', 'mean_compactness', 'mean_concavity',
        'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension'
    ]
    
    try:
        # Get mean values
        mean_values = [float(input_dict[feature]) for feature in mean_features]
        
        # Generate SE (standard error) values - typically 10-20% of mean values
        se_values = [val * 0.15 for val in mean_values]  # 15% of mean as SE
        
        # Generate worst values - typically 120-150% of mean values
        worst_values = [val * 1.35 for val in mean_values]  # 135% of mean as worst
        
        # Combine all 30 features: mean + se + worst
        all_features = mean_values + se_values + worst_values
        input_array = np.array(all_features).reshape(1, -1)
        
        logger.info(f"Input preprocessed successfully: shape {input_array.shape}")
        return input_array
    except Exception as e:
        logger.error(f"Error preprocessing input: {str(e)}")
        raise

def create_full_feature_array(mean_values: list) -> np.ndarray:
    """
    Create a full 30-feature array from 10 mean values by generating SE and worst features.
    
    Args:
        mean_values: List of 10 mean feature values
    
    Returns:
        NumPy array with shape (1, 30)
    """
    try:
        # Generate SE features (standard error) - typically smaller than mean
        se_values = [val * 0.15 for val in mean_values]  # 15% of mean
        
        # Generate worst features - typically larger than mean
        worst_values = [val * 1.35 for val in mean_values]  # 135% of mean
        
        # Combine all features
        all_features = mean_values + se_values + worst_values
        return np.array(all_features).reshape(1, -1)
    
    except Exception as e:
        logger.error(f"Error creating full feature array: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Load model and scaler
    model, scaler = load_model(model_path='model/svm_model.joblib', scaler_path='model/scaler.joblib')
    
    if model is None:
        logger.error("Failed to load model, cannot make prediction")
        raise ValueError("Model loading failed")

    # Example input dictionary (values from Wisconsin Breast Cancer dataset)
    input_dict = {
        'mean_radius': 17.99,
        'mean_texture': 10.38,
        'mean_perimeter': 122.80,
        'mean_area': 1001.0,
        'mean_smoothness': 0.11840,
        'mean_compactness': 0.27760,
        'mean_concavity': 0.3001,
        'mean_concave_points': 0.14710,
        'mean_symmetry': 0.2419,
        'mean_fractal_dimension': 0.07871
    }

    # Validate and preprocess input
    if validate_input(input_dict):
        input_array = preprocess_input(input_dict)
        # Make prediction
        prediction, confidence = make_prediction(model, input_array, scaler)
        print(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
    else:
        logger.error("Invalid input data, cannot make prediction")