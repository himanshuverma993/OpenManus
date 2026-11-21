import argparse
import asyncio

from app.agent.sandbox_agent import SandboxManus
from app.logger import logger
from app.utils.prompt import resolve_prompt


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Manus agent with a prompt")
    parser.add_argument(
        "--prompt", type=str, required=False, help="Input prompt for the agent"
    )
    args = parser.parse_args()

    prompt = resolve_prompt(args.prompt)
    if not prompt:
        logger.error(
            'No prompt supplied. Use `--prompt "your task"` or set the '
            "OPENMANUS_PROMPT environment variable."
        )
        return

    # Create and initialize Manus agent
    agent = await SandboxManus.create()
    try:
        logger.warning("Processing your request...")
        result = await agent.run(prompt)
        logger.info("Request processing completed.")
        print("\n=== FINAL RESULT ===")
        print(result)
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
