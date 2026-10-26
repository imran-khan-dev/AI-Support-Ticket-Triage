from fastapi import APIRouter

from app.modules.agent.route import router as AgentRoutes
from app.modules.analyze.route import router as AnalyzeRoutes
from app.modules.approvals.route import router as ApprovalsRoutes
from app.modules.integrations.route import router as IntegrationsRoutes

router = APIRouter()

module_routes = [
    {
        "path": "/analyze",
        "route": AnalyzeRoutes,
    },
    {
        "path": "/agent",
        "route": AgentRoutes,
    },
    {
        "path": "/approvals",
        "route": ApprovalsRoutes,
    },
    {
        "path": "/integrations",
        "route": IntegrationsRoutes,
    },
]

for module in module_routes:
    router.include_router(
        module["route"],
        prefix=module["path"],
    )
