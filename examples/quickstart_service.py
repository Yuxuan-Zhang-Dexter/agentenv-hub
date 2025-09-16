from __future__ import annotations

import uvicorn
from agentenv_hub.transports.agent_server import create_app


def main():
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()


