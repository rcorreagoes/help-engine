from pathlib import Path

import numpy as np
import pandas as pd

# Carregando o arquivo original
BASE_DIR = Path(__file__).resolve().parents[1]
input_file = BASE_DIR / 'data' / 'raw' / 'PLANILHA-HELP.ods'
df_orig = pd.read_excel(input_file, engine='odf')

# Lista para acumular os registros finais tratados
registros_ml = []

# Como os meses e locais estão distribuídos, vamos ler o arquivo linha por linha
# e identificar dinamicamente quando começa um novo bloco de mês/local.
mes_atual = "JANEIRO"
local_atual = "CAONZE"

# Pegamos a lista de encargos únicos padrão que queremos como colunas
colunas_custos = ['SALÁRIO', 'INSS / GPS', 'IRRF / FGTS', 'FERIAS / 13', 
                  'UNIFORME / EPI', 'ALIMENT', 'TRANSP', 'DEMISSÃO', 
                  'PLANO SAÚDE', 'EXAME OCUPAC', 'COMISSÃO']

# Vamos varrer o dataframe original identificando blocos
i = 0
while i < len(df_orig):
    linha_val = str(df_orig.iloc[i, 0]).strip().upper()
    
    # Detecta se é uma linha indicando nova folha/mês/local, ex: "FEVEREIRO MESQUITA"
    if any(m in linha_val for m in ["JANEIRO", "FEVEREIRO", "MARÇO", "ABRIL", "MAIO", "JUNHO", "JULHO", "AGOSTO", "SETEMBRO", "OUTUBRO", "NOVEMBRO", "DEZEMBRO"]):
        partes = linha_val.split()
        mes_atual = partes[0]
        if len(partes) > 1:
            local_atual = partes[1]
        i += 1
        continue
        
    # Se encontramos a linha de cargos/cabeçalho de colunas do bloco
    if "ENCARGOS" in linha_val or pd.isna(df_orig.iloc[i, 0]):
        # Mapeia cargos da linha atual (ex: GERENTE, 1, 2, AJUDANTE...)
        cargos_bloco = {}
        for col_idx in range(1, len(df_orig.columns)):
            cargo = df_orig.iloc[i, col_idx]
            if pd.notna(cargo) and str(cargo).strip().upper() != "ENCARGO":
                cargo_nome = str(cargo).strip()
                # Corrige typo AMD → ADM
                if cargo_nome.upper() == "AMD":
                    cargo_nome = "ADM"
                # Converte identificadores numéricos (1, 2, 3...) para nome descritivo
                try:
                    num = int(float(cargo_nome))
                    cargo_nome = f"FUNCIONÁRIO {num}"
                except (ValueError, TypeError):
                    pass
                cargos_bloco[col_idx] = cargo_nome
        
        # Processa as próximas linhas de custos deste bloco até o próximo cabeçalho
        dados_cargos = {c: {k: 0.0 for k in colunas_custos} for c in cargos_bloco.values()}
        
        i += 1
        while i < len(df_orig):
            proxima_linha_val = str(df_orig.iloc[i, 0]).strip().upper()
            # Se achar outro bloco de mês ou cabeçalho, para este loop
            if any(m in proxima_linha_val for m in ["JANEIRO", "FEVEREIRO", "MARÇO", "ABRIL", "MAIO", "JUNHO", "JULHO", "AGOSTO", "SETEMBRO", "OUTUBRO", "NOVEMBRO", "DEZEMBRO"]) or "ENCARGOS" in proxima_linha_val:
                i -= 1 # Devolve a linha para o loop externo
                break
            
            # Mapeia qual é a linha de encargo
            encargo_nome = None
            for col_c in colunas_custos:
                if col_c in proxima_linha_val:
                    encargo_nome = col_c
                    break
            
            if encargo_nome:
                # Preenche o valor de cada cargo nessa linha
                for col_idx, cargo in cargos_bloco.items():
                    # Valor absoluto do custo
                    val = df_orig.iloc[i, col_idx]
                    # Se for nulo, trata como 0.0
                    val_tratado = float(val) if pd.notna(val) and isinstance(val, (int, float)) else 0.0
                    dados_cargos[cargo][encargo_nome] = val_tratado
            i += 1
            
        # Transforma o dicionário do bloco em registros tabulares para o ML
        for cargo, custos in dados_cargos.items():
            registro = {
                "Mês": mes_atual,
                "Local": local_atual,
                "Cargo": cargo
            }
            registro.update(custos)
            registros_ml.append(registro)
            
    i += 1

# Cria o DataFrame consolidado
df_estruturado_ml = pd.DataFrame(registros_ml)

# Substitui possíveis NaNs restantes por 0
df_estruturado_ml.fillna(0, inplace=True)

# Remove registros onde todos os custos são zero E o cargo não é um funcionário numerado
# (linhas completamente vazias do Excel, não funcionários com custos zero)
mask_custos_zero = df_estruturado_ml[colunas_custos].sum(axis=1) == 0
mask_cargo_vazio = ~df_estruturado_ml['Cargo'].str.startswith('FUNCIONÁRIO')
df_estruturado_ml = df_estruturado_ml[~(mask_custos_zero & mask_cargo_vazio)]

# Salva em formato CSV
output_dir = BASE_DIR / 'data' / 'processed'
output_dir.mkdir(parents=True, exist_ok=True)
caminho_final_csv = output_dir / 'dados_prontos_para_ml.csv'
df_estruturado_ml.to_csv(caminho_final_csv, index=False, encoding='utf-8')

print(f"Estrutura criada com sucesso! Salva em formato CSV: {caminho_final_csv}")
