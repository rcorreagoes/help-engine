import os
from src.preprocessing import process_housing_data
from src.train import train_housing_model

def run_full_pipeline():
    print("🚀 [1/2] A iniciar o pré-processamento e engenharia de features...")
    process_housing_data(
        input_path="data/raw/portugal_housing.csv",
        output_clean_path="data/processed/housing_clean.csv",
        output_trends_path="data/processed/district_yearly_trends.csv"
    )
    
    print("\n🤖 [2/2] A iniciar o treinamento do modelo de Machine Learning...")
    train_housing_model()
    
    print("\n✨ Pipeline completo executado com sucesso! Sistema pronto para subir a API.")

if __name__ == "__main__":
    run_full_pipeline()