import argparse
import asyncio
import time

from app.agent.manus import Manus
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger
from app.utils.prompt import resolve_prompt


async def run_flow(prompt: str):
    agents = {
        "manus": Manus(),
    }
    if config.run_flow_config.use_data_analysis_agent:
        # Imported lazily: the data-analysis agent needs pandas / matplotlib,
        # which are not required for the plain Manus flow.
        from app.agent.data_analysis import DataAnalysis

        agents["data_analysis"] = DataAnalysis()
    try:
        if prompt.strip().isspace() or not prompt:
            logger.warning("Empty prompt provided.")
            return

        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )
        logger.warning("Processing your request...")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 60 minute timeout for the entire execution
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Request processed in {elapsed_time:.2f} seconds")
            logger.info(result)
            print("\n=== FINAL RESULT ===")
            print(result)
        except asyncio.TimeoutError:
            logger.error("Request processing timed out after 1 hour")
            logger.info(
                "Operation terminated due to timeout. Please try a simpler request."
            )

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the OpenManus multi-agent flow")
    parser.add_argument(
        "--prompt", type=str, required=False, help="Input prompt for the flow"
    )
    args = parser.parse_args()

    prompt = resolve_prompt(args.prompt)
    if not prompt:
        logger.error(
            'No prompt supplied. Use `--prompt "your task"` or set the '
            "OPENMANUS_PROMPT environment variable."
        )
    else:
        asyncio.run(run_flow(prompt))
