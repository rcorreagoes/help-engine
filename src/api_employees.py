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
    mes_col   = _get_existing_column(df, ['Mês', 'Mes'])
    local_col = _get_existing_column(df, ['Local'])
    salario_col  = _get_existing_column(df, ['SALÁRIO'])
    inss_col     = _get_existing_column(df, ['INSS / GPS'])
    irrf_col     = _get_existing_column(df, ['IRRF / FGTS'])
    uniforme_col = _get_existing_column(df, ['UNIFORME / EPI'])
    aliment_col  = _get_existing_column(df, ['ALIMENT'])
    transp_col   = _get_existing_column(df, ['TRANSP'])
    plano_col    = _get_existing_column(df, ['PLANO SAÚDE'])
    ferias_col   = _get_existing_column(df, ['FÉRIAS / 13º', 'FERIAS / 13'])
    demissao_col = _get_existing_column(df, ['DEMISSÃO', 'RESCISÃO', 'RESCISAO'])

    if local_filtro:
        df = df[df[local_col].astype(str).str.upper() == local_filtro.upper()]

    # --- Totais acumulados (todos os meses) ---
    total_salario    = df[salario_col].sum()
    total_impostos   = df[inss_col].sum() + df[irrf_col].sum()
    total_beneficios = df[[uniforme_col, aliment_col, transp_col, plano_col]].sum().sum()
    total_ferias_13  = df[ferias_col].sum()
    total_demissao   = df[demissao_col].sum()
    total_provisoes  = total_ferias_13 + total_demissao
    custo_total_geral = total_salario + total_impostos + total_beneficios + total_provisoes

    # --- Valores do último mês disponível ---
    # Ordem calendária dos meses em português
    ordem_meses = [
        'JANEIRO', 'FEVEREIRO', 'MARÇO', 'ABRIL', 'MAIO', 'JUNHO',
        'JULHO', 'AGOSTO', 'SETEMBRO', 'OUTUBRO', 'NOVEMBRO', 'DEZEMBRO'
    ]
    meses_disponiveis = df[mes_col].astype(str).str.upper().unique().tolist()
    meses_ordenados   = [m for m in ordem_meses if m in meses_disponiveis]
    ultimo_mes        = meses_ordenados[-1] if meses_ordenados else None

    df_ultimo_mes = df[df[mes_col].astype(str).str.upper() == ultimo_mes] if ultimo_mes else df.iloc[0:0]

    mes_salario    = df_ultimo_mes[salario_col].sum()
    mes_impostos   = df_ultimo_mes[inss_col].sum() + df_ultimo_mes[irrf_col].sum()
    mes_beneficios = df_ultimo_mes[[uniforme_col, aliment_col, transp_col, plano_col]].sum().sum()
    mes_ferias_13  = df_ultimo_mes[ferias_col].sum()
    mes_demissao   = df_ultimo_mes[demissao_col].sum()
    mes_provisoes  = mes_ferias_13 + mes_demissao
    mes_custo_total = mes_salario + mes_impostos + mes_beneficios + mes_provisoes

    return {
        "local": local_filtro or "TODOS",
        # Totais acumulados (mantidos sem alteração)
        "custo_total_folha": float(custo_total_geral),
        "impostos_totais": float(total_impostos),
        "beneficios_totais": float(total_beneficios),
        "provisoes_totais": {
            "ferias_13": float(total_ferias_13),
            "demissao_rescisao": float(total_demissao),
            "total": float(total_provisoes)
        },
        # Valores do último mês disponível
        "ultimo_mes": ultimo_mes,
        "custo_total_folha_mes": float(mes_custo_total),
        "impostos_totais_mes": float(mes_impostos),
        "beneficios_totais_mes": float(mes_beneficios),
        "provisoes_totais_mes": float(mes_provisoes),
    }

# --- 2. Custos por Cargo (com descritivo de impostos) ---
def get_custos_por_cargo(cargo_filtro=None):
    df = carregar_dados()
    mes_col = _get_existing_column(df, ['Mês', 'Mes'])
    cargo_col = _get_existing_column(df, ['Cargo'])
    salario_col = _get_existing_column(df, ['SALÁRIO'])
    inss_col = _get_existing_column(df, ['INSS / GPS'])
    irrf_col = _get_existing_column(df, ['IRRF / FGTS'])
    aliment_col = _get_existing_column(df, ['ALIMENT'])
    transp_col = _get_existing_column(df, ['TRANSP'])
    plano_col = _get_existing_column(df, ['PLANO SAÚDE'])

    ordem_meses = [
        'JANEIRO', 'FEVEREIRO', 'MARÇO', 'ABRIL', 'MAIO', 'JUNHO',
        'JULHO', 'AGOSTO', 'SETEMBRO', 'OUTUBRO', 'NOVEMBRO', 'DEZEMBRO'
    ]
    meses_disponiveis = df[mes_col].astype(str).str.upper().unique().tolist()
    meses_ordenados = [m for m in ordem_meses if m in meses_disponiveis]
    ultimo_mes = meses_ordenados[-1] if meses_ordenados else None

    if ultimo_mes:
        df = df[df[mes_col].astype(str).str.upper() == ultimo_mes]

    if cargo_filtro:
        df = df[df[cargo_col].astype(str).str.upper() == cargo_filtro.upper()]

    # O endpoint deve refletir apenas o último mês disponível, sem somar meses anteriores
    resultado = []
    for cargo, grupo in df.groupby(cargo_col, sort=True):
        salario_base = float(grupo[salario_col].sum())
        inss_total = float(grupo[inss_col].sum())
        irrf_total = float(grupo[irrf_col].sum())
        aliment_total = float(grupo[aliment_col].sum())
        transp_total = float(grupo[transp_col].sum())
        plano_total = float(grupo[plano_col].sum())

        resultado.append({
            "cargo": str(cargo),
            "salario_base": salario_base,
            "impostos_e_encargos": {
                "inss_gps": inss_total,
                "irrf_fgts": irrf_total
            },
            "beneficios": {
                "alimentacao": aliment_total,
                "transporte": transp_total,
                "plano_saude": plano_total
            },
            "custo_total_cargo": float(
                salario_base + inss_total + irrf_total + aliment_total + transp_total + plano_total
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