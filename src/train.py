import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_housing_model():
    processed_file = "data/processed/housing_clean.csv"
    model_output_path = "models/housing_model.pkl"
    columns_output_path = "models/model_columns.pkl"
    
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Ficheiro processado não encontrado em: {processed_file}. Execute o preprocessing.py primeiro.")

    print("📊 A carregar os dados limpos e processados...")
    df = pd.read_csv(processed_file)

    # 1. Definir a variável alvo (y) e as features (X)
    target_column = 'Price'
    if target_column not in df.columns:
        raise ValueError(f"A coluna alvo '{target_column}' não foi encontrada no dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"🔢 Número total de features (incluindo One-Hot Encoding): {X.shape[1]}")
    print(f"📈 Número de amostras para treino: {len(X)}")

    # 2. Dividir os dados em conjuntos de Treino (80%) e Teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🤖 A treinar o modelo de Machine Learning (RandomForestRegressor)...")
    model = RandomForestRegressor(
        n_estimators=150, 
        max_depth=25, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 3. Avaliar o modelo com os dados de teste
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n📊 Resultados da Avaliação do Modelo:")
    print(f"   - RMSE (Erro Médio): €{rmse:,.2f}")
    print(f"   - R² Score (Precisão): {r2:.4f} (quanto mais próximo de 1.0, melhor)")

    # 4. Salvar o modelo e a lista exata de colunas em artefatos binários
    os.makedirs("models", exist_ok=True)
    
    joblib.dump(model, model_output_path)
    # Salvamos a lista de colunas para garantir que a futura API alinhe os inputs corretamente
    joblib.dump(list(X.columns), columns_output_path)
    
    print(f"\n💾 Modelo salvo com sucesso em: {model_output_path}")
    print(f"💾 Estrutura de colunas salva em: {columns_output_path}")

if __name__ == "__main__":
    train_housing_model()

#++++++++++++++++++
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, VotingRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# import joblib
# import os

# def train_housing_model():
#     processed_file = "data/processed/housing_clean.csv"
#     model_output_path = "models/housing_model.pkl"
#     columns_output_path = "models/model_columns.pkl"
    
#     if not os.path.exists(processed_file):
#         raise FileNotFoundError(f"Ficheiro processado não encontrado em: {processed_file}. Execute o preprocessing.py primeiro.")

#     print("📊 A carregar os dados processados...")
#     df = pd.read_csv(processed_file)

#     # 1. Definir a variável alvo (y) e as features (X)
#     target_column = 'Price'
#     if target_column not in df.columns:
#         raise ValueError(f"A coluna alvo '{target_column}' não foi encontrada no dataset.")

#     X = df.drop(columns=[target_column])
#     y = df[target_column]

#     print(f"🔢 Número total de features: {X.shape[1]}")
#     print(f"📈 Número de amostras para treino: {len(X)}")

#     # 2. Dividir os dados em Treino (80%) e Teste (20%)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     print("🤖 A instanciar os modelos base para o Ensemble...")
    
#     # Modelo 1: Random Forest
#     rf_model = RandomForestRegressor(
#         n_estimators=150, 
#         max_depth=22, 
#         random_state=42, 
#         n_jobs=-1
#     )
    
#     # Modelo 2: XGBoost Otimizado
#     xgb_model = XGBRegressor(
#         n_estimators=400,
#         learning_rate=0.04,
#         max_depth=7,
#         subsample=0.85,
#         colsample_bytree=0.85,
#         reg_alpha=0.1,
#         reg_lambda=1.0,
#         random_state=42,
#         n_jobs=-1
#     )

#     # 3. Criar o Voting Regressor (Ensemble)
#     # Combinamos os dois com pesos iguais (ou você pode dar mais peso ao XGBoost, ex: weights=[1, 2])
#     print("🤝 A construir o Ensemble (Random Forest + XGBoost)...")
#     ensemble_model = VotingRegressor(
#         estimators=[
#             ('rf', rf_model),
#             ('xgb', xgb_model)
#         ],
#         weights=[1, 1.5] # Damos um peso ligeiramente superior ao XGBoost
#     )

#     print("🚀 A treinar o Ensemble Regressor...")
#     ensemble_model.fit(X_train, y_train)

#     # 4. Avaliar o modelo unificado com os dados de teste
#     predictions = ensemble_model.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, predictions)

#     print("\n📊 Resultados da Avaliação do Ensemble:")
#     print(f"   - RMSE (Erro Médio): €{rmse:,.2f}")
#     print(f"   - R² Score (Precisão): {r2:.4f} (quanto mais próximo de 1.0, melhor)")

#     # 5. Salvar o modelo combinado e as colunas
#     os.makedirs("models", exist_ok=True)
    
#     joblib.dump(ensemble_model, model_output_path)
#     joblib.dump(list(X.columns), columns_output_path)
    
#     print(f"\n💾 Modelo Ensemble salvo com sucesso em: {model_output_path}")
#     print(f"💾 Estrutura de colunas salva em: {columns_output_path}")

# if __name__ == "__main__":
#     train_housing_model()