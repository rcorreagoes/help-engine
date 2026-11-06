from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.api_employees import router as employees_router
# Descomente conforme for implementando os outros módulos:
# from src.api_costs import router as costs_router
# from src.api_invoicing import router as invoicing_router
# from src.api_betting_slips import router as betting_slips_router

app = FastAPI(
    title="API Global - Help Engine",
    description="Central de endpoints de custos, colaboradores e relatórios analíticos",
    version="1.0.0"
)

# Configuração de CORS (permite que o frontend Angular e backend NestJS se comuniquem com a API)
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

# Inclusão dos roteadores por domínio
app.include_router(employees_router, prefix="/api/v1")
# app.include_router(costs_router, prefix="/api/v1")
# app.include_router(invoicing_router, prefix="/api/v1")
# app.include_router(betting_slips_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
