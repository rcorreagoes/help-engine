import os

import numpy as np
import pandas as pd


def process_housing_data(input_path: str, output_clean_path: str, output_trends_path: str):
    print("📥 A carregar o dataset bruto...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Ficheiro não encontrado em: {input_path}")
        
    df = pd.read_csv(input_path)
    print(f"📊 Linhas originais: {len(df)}")

    # 1. Tratamentos iniciais de nulos
    if 'Town' in df.columns:
        df['Town'] = df['Town'].fillna(df['Town'].mode()[0] if not df['Town'].mode().empty else 'Unknown')
    if 'Type' in df.columns:
        df['Type'] = df['Type'].fillna(df['Type'].mode()[0] if not df['Type'].mode().empty else 'Unknown')
    for col in ['EnergyCertificate', 'EnergyEfficiencyLevel', 'ConservationStatus']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    colunas_numericas = ['NumberOfBathrooms', 'TotalArea', 'LivingArea', 'TotalRooms', 'NumberOfBedrooms', 'NumberOfWC', 'BuiltArea']
    for coluna in colunas_numericas:
        if coluna in df.columns:
            df[coluna] = df[coluna].fillna(df[coluna].median())

    df = df.dropna(subset=['Price'])

    if 'Floor' in df.columns:
        df['Floor'] = df['Floor'].fillna('Unknown')
        
    if 'ConstructionYear' in df.columns and 'Type' in df.columns:
        df['ConstructionYear'] = df.groupby('Type')['ConstructionYear'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else 2000)
        )
        
    if 'Parking' in df.columns:
        df['Parking'] = df['Parking'].fillna(df['Parking'].mode()[0] if not df['Parking'].mode().empty else 'Unknown')
        
    for col in ['Garage', 'ElectricCarsCharging', 'Elevator']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)

    # 2. Tratamento de Datas e Anos
    if 'PublishDate' in df.columns:
        df['PublishDate'] = pd.to_datetime(df['PublishDate'], errors='coerce')
        df['PublishDate_missing'] = df['PublishDate'].isna().astype(int)
        df['PublishDate'] = df['PublishDate'].fillna(pd.Timestamp('1970-01-01'))
        df['PublishYear'] = df['PublishDate'].dt.year
        df['PublishMonth'] = df['PublishDate'].dt.month
        df = df.drop(columns=['PublishDate'])

    if 'LotSize' in df.columns:
        df['LotSize_missing'] = df['LotSize'].isna().astype(int)
        df['LotSize'] = df['LotSize'].fillna(0)

    # 3. Limpeza de Outliers baseada em Preço e Área
    if 'Price' in df.columns and 'TotalArea' in df.columns:
        df['PricePerSqm'] = df['Price'] / df['TotalArea']
        q_low_p, q_high_p = df['Price'].quantile(0.01), df['Price'].quantile(0.99)
        q_low_sqm, q_high_sqm = df['TotalArea'].quantile(0.01), df['TotalArea'].quantile(0.99)

        df = df[
            (df['Price'] >= q_low_p) & (df['Price'] <= q_high_p) &
            (df['TotalArea'] >= q_low_sqm) & (df['TotalArea'] <= q_high_sqm)
        ].copy()

# =========================================================================
    # ARTEFATO 1: Tabela de Tendências Históricas por Distrito, Município, Tipo e Ano
    # =========================================================================
    municipality_col = 'City' if 'City' in df.columns else ('Town' if 'Town' in df.columns else None)

    if 'PublishYear' in df.columns and 'District' in df.columns and municipality_col and 'Type' in df.columns:
        print(f"📈 A gerar tendências históricas por Distrito, {municipality_col}, Tipo de Imóvel e Ano...")
        df_valid_years = df[df['PublishYear'] >= 2018]
        
        # Agrupar incluindo o 'Type'
        trends_df = df_valid_years.groupby(['District', municipality_col, 'Type', 'PublishYear'])['PricePerSqm'].median().reset_index()
        
        # Padronizar nomes das colunas no CSV final
        trends_df.rename(columns={
            'PricePerSqm': 'MedianPricePerSqm',
            municipality_col: 'Municipality'
        }, inplace=True)
        
        os.makedirs(os.path.dirname(output_trends_path), exist_ok=True)
        trends_df.to_csv(output_trends_path, index=False)
        print(f"✨ Tabela de tendências multidimensional salva em: {output_trends_path}")

    # =========================================================================
    # ARTEFATO 2: Base Filtrada para Previsão Atual (Focada em anúncios recentes)
    # =========================================================================
    print("🎯 A preparar base de dados para o modelo de valor atual (2025/2026)...")
    # Mantemos preferencialmente anúncios recentes para o modelo refletir o mercado de hoje
    df_current = df[df['PublishYear'] >= 2023].copy()
    if len(df_current) < 1000: # Fallback de segurança caso o dataset tenha poucos dados recentes
        df_current = df.copy()

    # Criar features avançadas para o modelo atual
    current_year = 2026
    df_current['PropertyAge'] = (current_year - df_current['ConstructionYear']).apply(lambda x: max(0, x))
    
    district_median_sqm = df_current.groupby('District')['PricePerSqm'].median().to_dict()
    df_current['DistrictPricePerSqm'] = df_current['District'].map(district_median_sqm).fillna(df_current['PricePerSqm'].median())
    
    # Remover coluna auxiliar temporária
    df_current = df_current.drop(columns=['PricePerSqm'])

    # One-Hot Encoding
    categorical_cols = [
        'District', 'City', 'Town', 'Type', 'EnergyCertificate', 
        'EnergyEfficiencyLevel', 'ConservationStatus', 'Floor', 
        'Parking', 'Garage', 'ElectricCarsCharging', 'Elevator'
    ]
    categorical_cols_present = [col for col in categorical_cols if col in df_current.columns]
    
    print("🔄 A aplicar One-Hot Encoding na base atual...")
    df_encoded = pd.get_dummies(df_current, columns=categorical_cols_present, drop_first=True)

    # Salvar base limpa atual
    os.makedirs(os.path.dirname(output_clean_path), exist_ok=True)
    df_encoded.to_csv(output_clean_path, index=False)
    print(f"✨ Base limpa atual salva em: {output_clean_path}")

    return df_encoded

if __name__ == "__main__":
    process_housing_data(
        input_path="data/raw/portugal_housing.csv",
        output_clean_path="data/processed/housing_clean.csv",
        output_trends_path="data/processed/district_yearly_trends.csv"
    )