from placement_model import train_models, save_artifacts


def main():
    artifacts = train_models()
    save_artifacts(artifacts['model'], artifacts['scaler'], artifacts['encoders'])

    print('Model training completed and artifacts saved.')
    print('Accuracy:', artifacts['metrics']['accuracy'])
    print('ROC AUC:', artifacts['metrics']['roc_auc'])
    print('\nClassification Report:')
    print(artifacts['metrics']['classification_report'])
    print('\nConfusion Matrix:')
    print(artifacts['metrics']['confusion_matrix'])


if __name__ == '__main__':
    main()
