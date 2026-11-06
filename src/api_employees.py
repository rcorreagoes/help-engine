from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Caminho configurado para o arquivo CSV processado
BASE_DIR = Path(__file__).resolve().parents[1]
output_dir = BASE_DIR / 'data' / 'processed'
caminho_final_csv = output_dir / 'dados_prontos_para_ml.csv'

router = APIRouter(prefix="/employees", tags=["Employees"])


def _get_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(f"Nenhuma das colunas esperadas foi encontrada: {candidates}")


def carregar_dados():
    if not caminho_final_csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado em: {caminho_final_csv}")

    df = pd.read_csv(caminho_final_csv)

    salario_col = _get_existing_column(df, ['SALÁRIO'])
    inss_col = _get_existing_column(df, ['INSS / GPS'])
    irrf_col = _get_existing_column(df, ['IRRF / FGTS'])
    ferias_col = _get_existing_column(df, ['FÉRIAS / 13º', 'FERIAS / 13'])
    uniforme_col = _get_existing_column(df, ['UNIFORME / EPI'])
    aliment_col = _get_existing_column(df, ['ALIMENT'])
    transp_col = _get_existing_column(df, ['TRANSP'])
    demissao_col = _get_existing_column(df, ['DEMISSÃO', 'DEMISSÃO'])
    plano_col = _get_existing_column(df, ['PLANO SAÚDE'])

    # Garantir que as colunas numéricas estão limpas e no formato float
    colunas_custo = [
        salario_col, inss_col, irrf_col, ferias_col,
        uniforme_col, aliment_col, transp_col, demissao_col, plano_col
    ]
    for col in colunas_custo:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    return df


# --- 1. Custos Totais ---
def get_custos_totais(local_filtro=None):
    df = carregar_dados()
    
    # Mapeamento de colunas existentes
    local_col = _get_existing_column(df, ['Local'])
    salario_col = _get_existing_column(df, ['SALÁRIO'])
    inss_col = _get_existing_column(df, ['INSS / GPS'])
    irrf_col = _get_existing_column(df, ['IRRF / FGTS'])
    uniforme_col = _get_existing_column(df, ['UNIFORME / EPI'])
    aliment_col = _get_existing_column(df, ['ALIMENT'])
    transp_col = _get_existing_column(df, ['TRANSP'])
    plano_col = _get_existing_column(df, ['PLANO SAÚDE'])
    
    # Novas colunas de provisão
    ferias_col = _get_existing_column(df, ['FÉRIAS / 13º', 'FERIAS / 13'])
    demissao_col = _get_existing_column(df, ['DEMISSÃO', 'RESCISÃO', 'RESCISAO'])

    if local_filtro:
        df = df[df[local_col].astype(str).str.upper() == local_filtro.upper()]

    total_salario = df[salario_col].sum()
    total_impostos = df[inss_col].sum() + df[irrf_col].sum()
    total_beneficios = df[[uniforme_col, aliment_col, transp_col, plano_col]].sum().sum()
    
    # Totais de Provisões
    total_ferias_13 = df[ferias_col].sum()
    total_demissao = df[demissao_col].sum()
    total_provisoes = total_ferias_13 + total_demissao

    custo_total_geral = total_salario + total_impostos + total_beneficios + total_provisoes

    return {
        "local": local_filtro or "TODOS",
        "custo_total_folha": float(custo_total_geral),
        "impostos_totais": float(total_impostos),
        "beneficios_totais": float(total_beneficios),
        "provisoes_totais": {
            "ferias_13": float(total_ferias_13),
            "demissao_rescisao": float(total_demissao),
            "total": float(total_provisoes)
        }
    }

# --- 2. Custos por Cargo (com descritivo de impostos) ---
def get_custos_por_cargo(cargo_filtro=None):
    df = carregar_dados()
    cargo_col = _get_existing_column(df, ['Cargo'])
    salario_col = _get_existing_column(df, ['SALÁRIO'])
    inss_col = _get_existing_column(df, ['INSS / GPS'])
    irrf_col = _get_existing_column(df, ['IRRF / FGTS'])
    aliment_col = _get_existing_column(df, ['ALIMENT'])
    transp_col = _get_existing_column(df, ['TRANSP'])
    plano_col = _get_existing_column(df, ['PLANO SAÚDE'])

    if cargo_filtro:
        df = df[df[cargo_col].astype(str).str.upper() == cargo_filtro.upper()]

    # Agrupa por cargo somando os valores
    agrupado = df.groupby(cargo_col).agg({
        salario_col: 'sum',
        inss_col: 'sum',
        irrf_col: 'sum',
        aliment_col: 'sum',
        transp_col: 'sum',
        plano_col: 'sum'
    }).reset_index()

    resultado = []
    for _, row in agrupado.iterrows():
        resultado.append({
            "cargo": row[cargo_col],
            "salario_base": float(row[salario_col]),
            "impostos_e_encargos": {
                "inss_gps": float(row[inss_col]),
                "irrf_fgts": float(row[irrf_col])
            },
            "beneficios": {
                "alimentacao": float(row[aliment_col]),
                "transporte": float(row[transp_col]),
                "plano_saude": float(row[plano_col])
            },
            "custo_total_cargo": float(
                row[salario_col] + row[inss_col] + row[irrf_col] + row[aliment_col] + row[transp_col] + row[plano_col]
            )
        })
    return resultado


# --- 3. Valores de Férias, 13º e Demissão (Total e por Funcionário) ---
def get_provisoes_e_demissoes(local_filtro=None):
    df = carregar_dados()
    mes_col = _get_existing_column(df, ['Mês', 'Mes'])
    local_col = _get_existing_column(df, ['Local'])
    cargo_col = _get_existing_column(df, ['Cargo'])
    ferias_col = _get_existing_column(df, ['FÉRIAS / 13º', 'FERIAS / 13'])
    demissao_col = _get_existing_column(df, ['DEMISSÃO'])

    if local_filtro:
        df = df[df[local_col].astype(str).str.upper() == local_filtro.upper()]

    # Totais consolidados da empresa/local
    totais = {
        "total_ferias_13": float(df[ferias_col].sum()),
        "total_demissao": float(df[demissao_col].sum()),
        "total_geral_provisoes": float(df[ferias_col].sum() + df[demissao_col].sum())
    }

    # Detalhado por registo/funcionário
    detalhado = []
    for idx, row in df.iterrows():
        detalhado.append({
            "id": int(idx),
            "mes": row.get(mes_col, ''),
            "local": row.get(local_col, ''),
            "cargo": row.get(cargo_col, ''),
            "ferias_13": float(row[ferias_col]),
            "demissao": float(row[demissao_col]),
            "total_linha": float(row[ferias_col] + row[demissao_col])
        })

    return {
        "totais": totais,
        "detalhado_por_funcionario": detalhado
    }


@router.get("/custos-totais")
def custos_totais(local: Optional[str] = None):
    try:
        return get_custos_totais(local)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/custos-por-cargo")
def custos_por_cargo(cargo: Optional[str] = None):
    try:
        return get_custos_por_cargo(cargo)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/provisoes-demissoes")
def provisoes_demissoes(local: Optional[str] = None):
    try:
        return get_provisoes_e_demissoes(local)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app = FastAPI(
    title="API Employees - Help Engine",
    description="Endpoints de custos e provisões de colaboradores",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


app.include_router(router, prefix="/api/v1")