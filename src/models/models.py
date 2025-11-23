"""
"""Machine Learning models for baseball prospect projections
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import joblib
from pathlib import Path

# Constants
NOT_TRAINED_ERROR = "Model not trained"

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseModel:
    """
    Base class for all projection models
    """

    def __init__(self, model_name: str):
        """
        Initialize base model

        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        logger.info(f"Initialized {model_name}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model"""
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save model to disk"""
        if self.model is not None:
            joblib.dump(self.model, path)
            logger.info(f"Saved {self.model_name} to {path}")

    def load(self, path: str) -> None:
        """Load model from disk"""
        self.model = joblib.load(path)
        self.is_trained = True
        logger.info(f"Loaded {self.model_name} from {path}")


class RandomForestModel(BaseModel):
    """
    Random Forest model for projections
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Random Forest model

        Args:
            config: Configuration dictionary
        """
        super().__init__("RandomForest")

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available")

        self.config = config or {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": 42,
        }

    def train(self, X: pd.DataFrame, y: pd.Series, task: str = "regression") -> None:
        """
        Train Random Forest model

        Args:
            X: Training features
            y: Training targets
            task: 'regression' or 'classification'
        """
        logger.info(f"Training Random Forest for {task} with {len(X)} samples")

        if task == "regression":
            self.model = RandomForestRegressor(**self.config)
        else:
            self.model = RandomForestClassifier(**self.config)

        self.model.fit(X, y)
        self.is_trained = True

        logger.info("Random Forest training complete")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features for prediction

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance

        Returns:
            Series of feature importances
        """
        if not self.is_trained:
            raise ValueError(NOT_TRAINED_ERROR)

        return pd.Series(
            self.model.feature_importances_, index=self.model.feature_names_in_
        ).sort_values(ascending=False)


class XGBoostModel(BaseModel):
    """
    XGBoost model for projections
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize XGBoost model

        Args:
            config: Configuration dictionary
        """
        super().__init__("XGBoost")

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")

        self.config = config or {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
        }

    def train(
        self, X: pd.DataFrame, y: pd.Series, eval_set: Optional[Tuple] = None
    ) -> None:
        """
        Train XGBoost model

        Args:
            X: Training features
            y: Training targets
            eval_set: Optional validation set (X_val, y_val)
        """
        logger.info(f"Training XGBoost with {len(X)} samples")

        self.model = xgb.XGBRegressor(**self.config)

        if eval_set:
            self.model.fit(X, y, eval_set=[eval_set], verbose=False)
        else:
            self.model.fit(X, y)

        self.is_trained = True
        logger.info("XGBoost training complete")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features for prediction

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError(NOT_TRAINED_ERROR)

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance

        Returns:
            Series of feature importances
        """
        if not self.is_trained:
            raise ValueError(NOT_TRAINED_ERROR)

        importance_dict = self.model.get_booster().get_score(importance_type="weight")
        return pd.Series(importance_dict).sort_values(ascending=False)


class LightGBMModel(BaseModel):
    """
    LightGBM model for projections
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LightGBM model

        Args:
            config: Configuration dictionary
        """
        super().__init__("LightGBM")

        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")

        self.config = config or {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        }

    def train(
        self, X: pd.DataFrame, y: pd.Series, eval_set: Optional[Tuple] = None
    ) -> None:
        """
        Train LightGBM model

        Args:
            X: Training features
            y: Training targets
            eval_set: Optional validation set (X_val, y_val)
        """
        logger.info(f"Training LightGBM with {len(X)} samples")

        self.model = lgb.LGBMRegressor(**self.config)

        if eval_set:
            self.model.fit(
                X,
                y,
                eval_set=[eval_set],
                callbacks=[lgb.early_stopping(stopping_rounds=50)],
            )
        else:
            self.model.fit(X, y)

        self.is_trained = True
        logger.info("LightGBM training complete")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features for prediction

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError(NOT_TRAINED_ERROR)

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance

        Returns:
            Series of feature importances
        """
        if not self.is_trained:
            raise ValueError(NOT_TRAINED_ERROR)

        return pd.Series(
            self.model.feature_importances_, index=self.model.feature_name_
        ).sort_values(ascending=False)


class DeepLearningModel(BaseModel):
    """
    Deep learning model using PyTorch
    """

    def __init__(self, input_dim: int, config: Optional[Dict] = None):
        """
        Initialize deep learning model

        Args:
            input_dim: Number of input features
            config: Configuration dictionary
        """
        super().__init__("DeepLearning")

        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        self.config = config or {
            "hidden_layers": [256, 128, 64],
            "dropout": 0.3,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "early_stopping_patience": 10,
        }

        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        """
        Build neural network architecture

        Returns:
            PyTorch model
        """
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in self.config["hidden_layers"]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(self.config["dropout"]))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        model = nn.Sequential(*layers)
        return model.to(self.device)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """
        Train deep learning model

        Args:
            X: Training features
            y: Training targets
            x_val: Validation features
            y_val: Validation targets
        """
        logger.info(f"Training Deep Learning model with {len(X)} samples")

        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        y_tensor = torch.FloatTensor(y.values).reshape(-1, 1).to(self.device)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=0
        )

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=1e-5
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config["epochs"]):
            self.model.train()
            epoch_loss = 0

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            # Validation
            if x_val is not None and y_val is not None:
                val_loss = self._validate(x_val, y_val, criterion)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config["early_stopping_patience"]:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}"
                    )
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")

        self.is_trained = True
        logger.info("Deep Learning training complete")

    def _validate(self, x_val: pd.DataFrame, y_val: pd.Series, criterion) -> float:
        """
        Validate model

        Args:
            x_val: Validation features
            y_val: Validation targets
            criterion: Loss function

        Returns:
            Validation loss
        """
        self.model.eval()

        x_val_tensor = torch.FloatTensor(x_val.values).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(self.device)

        with torch.no_grad():
            outputs = self.model(x_val_tensor)
            val_loss = criterion(outputs, y_val_tensor).item()

        return val_loss

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features for prediction

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError(NOT_TRAINED_ERROR)

        self.model.eval()
        x_tensor = torch.FloatTensor(X.values).to(self.device)

        with torch.no_grad():
            predictions = self.model(x_tensor).cpu().numpy()

        return predictions.cpu().numpy().flatten()


class EnsembleModel:
    """
    Ensemble multiple models for better predictions
    """

    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        """
        Initialize ensemble model

        Args:
            models: List of trained models
            weights: Optional weights for each model
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)

        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

        logger.info(f"Initialized ensemble with {len(models)} models")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions

        Args:
            X: Features for prediction

        Returns:
            Weighted ensemble predictions
        """
        predictions = []

        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)

    def predict_with_uncertainty(
        self, X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates

        Args:
            X: Features for prediction

        Returns:
            Tuple of (predictions, uncertainty)
        """
        all_predictions = []

        for model in self.models:
            pred = model.predict(X)
            all_predictions.append(pred)

        all_predictions = np.array(all_predictions)

        # Mean prediction
        mean_pred = np.average(all_predictions, axis=0, weights=self.weights)

        # Standard deviation as uncertainty
        std_pred = np.std(all_predictions, axis=0)

        return mean_pred, std_pred


class ModelTrainer:
    """
    Trainer for managing multiple models and targets
    """

    def __init__(self, config: Dict):
        """
        Initialize model trainer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        logger.info("Initialized ModelTrainer")

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, BaseModel]]:
        """
        Train models for all targets

        Args:
            X_train: Training features
            y_train: Training targets (multiple columns)
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Dictionary of trained models by target
        """
        logger.info(f"Training models for {len(y_train.columns)} targets")

        for target in y_train.columns:
            logger.info(f"Training models for target: {target}")
            self.models[target] = {}

            y_target_train = y_train[target]
            y_target_val = y_val[target] if y_val is not None else None

            # Train Random Forest
            if "random_forest" in self.config.get("active_models", []):
                rf_model = RandomForestModel(self.config.get("random_forest"))
                rf_model.train(X_train, y_target_train)
                self.models[target]["random_forest"] = rf_model

            # Train XGBoost
            if "xgboost" in self.config.get("active_models", []):
                xgb_model = XGBoostModel(self.config.get("xgboost"))
                eval_set = (x_val, y_target_val) if x_val is not None else None
                xgb_model.train(X_train, y_target_train, eval_set)
                self.models[target]["xgboost"] = xgb_model

            # Train LightGBM
            if "lightgbm" in self.config.get("active_models", []):
                lgb_model = LightGBMModel(self.config.get("lightgbm"))
                eval_set = (x_val, y_target_val) if x_val is not None else None
                lgb_model.train(X_train, y_target_train, eval_set)
                self.models[target]["lightgbm"] = lgb_model

            # Train Deep Learning
            if "deep_learning" in self.config.get("active_models", []):
                dl_model = DeepLearningModel(
                    input_dim=X_train.shape[1], config=self.config.get("deep_learning")
                )
                dl_model.train(X_train, y_target_train, x_val, y_target_val)
                self.models[target]["deep_learning"] = dl_model

        logger.info("Model training complete")
        return self.models

    def create_ensembles(self) -> Dict[str, EnsembleModel]:
        """
        Create ensemble models for each target

        Returns:
            Dictionary of ensemble models
        """
        ensembles = {}

        for target, target_models in self.models.items():
            model_list = list(target_models.values())
            ensembles[target] = EnsembleModel(model_list)

        logger.info(f"Created ensembles for {len(ensembles)} targets")
        return ensembles

    def save_models(self, save_dir: str) -> None:
        """
        Save all trained models

        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for target, target_models in self.models.items():
            target_dir = save_path / target
            target_dir.mkdir(exist_ok=True)

            for model_name, model in target_models.items():
                model_file = target_dir / f"{model_name}.pkl"
                model.save(str(model_file))

        logger.info(f"Saved all models to {save_dir}")
