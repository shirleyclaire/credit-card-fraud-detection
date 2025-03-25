from data.data_loader import DataLoader
from features.preprocessor import Preprocessor
from models.isolation_forest import FraudDetector
from evaluation.metrics import ModelEvaluator  # Changed from evaluate_model
from analysis.exploratory_analysis import ExploratoryAnalysis
from sklearn.model_selection import train_test_split
from models.threshold_optimizer import ThresholdOptimizer, CostConfig
from interpretation.model_interpreter import ModelInterpreter

def main():
    # Initialize components
    data_loader = DataLoader('./data/creditcard.csv')  # Keep this path
    preprocessor = Preprocessor()
    
    # Load and split data
    df = data_loader.load_data()
    data_splits = data_loader.split_data(df)
    X_train, X_test, y_train, y_test = data_splits['train_test']
    
    # Perform EDA
    eda = ExploratoryAnalysis(df)
    eda.show_class_distribution()
    eda.analyze_feature_distributions()
    eda.analyze_pca_components()
    eda.generate_summary_stats()
    eda.analyze_time_amount_patterns()
    
    # Fit preprocessor on training data and transform both sets
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Split validation set for threshold optimization
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # Initialize and train model
    model = FraudDetector(
        contamination=len(y_train[y_train==1])/len(y_train),
        n_estimators=200,
        max_samples='auto',
        max_features=0.8,
        bootstrap=True
    )
    model.fit(X_train_processed)
    
    # Initialize model evaluator (using the class instead of function)
    evaluator = ModelEvaluator(model, X_train_processed, y_train)
    
    # Perform cross-validation
    cv_results = evaluator.cross_validate()
    print("\nCross-validation Results:")
    print(cv_results.describe())
    
    # Initialize threshold optimizer with custom cost configuration
    cost_config = CostConfig(
        false_positive_cost=10.0,
        false_negative_cost=100.0,
        max_review_capacity=100,
        target_precision=0.95
    )
    threshold_optimizer = ThresholdOptimizer(cost_config)
    
    # Get optimal thresholds
    threshold_metrics = threshold_optimizer.optimize_threshold(
        y_val, 
        model.decision_function(X_val_processed),
        transaction_amounts=X_val['Amount']
    )
    
    # Get recommendation
    recommendation = threshold_optimizer.generate_recommendation(threshold_metrics)
    print("\nThreshold Recommendations:")
    for strategy, details in recommendation.items():
        print(f"\n{strategy.replace('_', ' ').title()}:")
        print(f"Threshold: {details['threshold']:.3f}")
        print(f"Explanation: {details['explanation']}")
    
    # Use cost-optimal threshold for final predictions
    model.threshold = threshold_metrics['cost_optimal']
    
    # Get predictions and scores
    y_scores = model.decision_function(X_test_processed)
    y_pred = model.predict(X_test_processed)
    
    # Comprehensive evaluation
    evaluation_results = evaluator.evaluate_model(y_test, y_pred, y_scores)
    print("\nEvaluation Results:")
    print(f"AUPRC: {evaluation_results['auprc']:.3f}")
    print(f"Baseline: {evaluation_results['baseline']:.3f}")
    print("\nClassification Report:")
    print(evaluation_results['classification_report'])
    
    # Feature importance analysis
    feature_importance = evaluator.feature_importance_analysis(
        X_test_processed.columns
    )
    print("\nTop 10 Most Important Features:")
    print(feature_importance.nlargest(10, 'importance'))
    
    # Get anomaly factors for analysis
    anomaly_factors = model.get_anomaly_factors(X_test_processed)
    
    # Save model
    model.save_model('models/fraud_detector.joblib')
    
    # Initialize model interpreter
    interpreter = ModelInterpreter(model, X_train_processed, X_train_processed.columns)
    
    # Generate interpretability insights
    print("\nGenerating model interpretability insights...")
    
    # Extract and display decision rules
    rules = interpreter.extract_decision_rules()
    print("\nKey Decision Rules:")
    print(rules)
    
    # Generate fraud prototypes
    prototypes = interpreter.generate_prototypes()
    print("\nFraud Prototypes Generated")
    
    # Create interactive dashboard
    app = interpreter.create_dashboard()
    print("\nStarting interpretation dashboard...")
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
