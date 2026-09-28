import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass 
class ModelTrainerConfig:
    trained_model_filepath = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test data")
            X_train, y_train, X_test, y_test= (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])
            
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier()
                
            }
            
            params = {
                "Decision Tree" : {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest" :{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],  #how many features to consider at each split
                    'n_estimators': [8,16,32,64,128,256]    #num of trees in the forest
                },
                "Gradient Boosting" : {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],      #scales how much each tree contributes
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],#fraction of training rows used to fit each tree
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'], 
                    'n_estimators': [8,16,32,64,128,256]     #number of trees 
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],        #step-size shrinkage per boosting round
                    'n_estimators': [8,16,32,64,128,256]      #number of boosting rounds
                },
                "CatBoosting Regressor":{ 
                    'depth': [6,8,10],                        #depth of each tree
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]               #number of trees
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],        #shrinks the contribution of each weak learner
                    # 'loss':['linear','square','exponential'], #how sample weights are updated after each round
                    'n_estimators': [8,16,32,64,128,256]      #number of weak learners
                }
                
            }
            
            model_report : dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,models=models, param=params)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("no best model found")
            
            logging.info(f"Best found model on both training and testing datasets: {best_model_name}")
            
            save_object(
                file_path= self.model_trainer_config.trained_model_filepath,
                obj= best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_sqr = r2_score(y_test, predicted)
            return r2_sqr
        except Exception as e:
            raise CustomException(e,sys)
