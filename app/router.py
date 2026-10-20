
from fastapi import APIRouter

from app.modules.agent.route import router as AgentRoutes
from app.modules.analyze.route import router as AnalyzeRoutes

router = APIRouter()

module_routes = [
    {
        "path": "/analyze",
        "route": AnalyzeRoutes,
    },
    {
        "path": "/agent",
        "route": AgentRoutes,
    }
]

for module in module_routes:
    router.include_router(
        module["route"],
        prefix=module["path"],
    )
