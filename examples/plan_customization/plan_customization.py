"""
Plan Customization Example

This example demonstrates how to use step callbacks to interrupt the agent after
plan creation, allow user interaction to approve or modify the plan, and then
resume execution while preserving agent memory.

Key concepts demonstrated:
1. Step callbacks to interrupt after PlanningStep
2. Extracting and modifying the current plan
3. Resuming execution with reset=False to preserve memory
4. User interaction for plan approval/modification
"""

from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, PlanningStep


def display_plan(plan_content):
    """Display the plan in a formatted way"""
    print("\n" + "=" * 60)
    print("🤖 AGENT PLAN CREATED")
    print("=" * 60)
    print(plan_content)
    print("=" * 60)


def get_user_choice():
    """Get user's choice for plan approval"""
    while True:
        choice = input("\nChoose an option:\n1. Approve plan\n2. Modify plan\n3. Cancel\nYour choice (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, or 3.")


def get_modified_plan(original_plan):
    """Allow user to modify the plan"""
    print("\n" + "-" * 40)
    print("MODIFY PLAN")
    print("-" * 40)
    print("Current plan:")
    print(original_plan)
    print("-" * 40)
    print("Enter your modified plan (press Enter twice to finish):")

    lines = []
    empty_line_count = 0

    while empty_line_count < 2:
        line = input()
        if line.strip() == "":
            empty_line_count += 1
        else:
            empty_line_count = 0
        lines.append(line)

    # Remove the last two empty lines
    modified_plan = "\n".join(lines[:-2])
    return modified_plan if modified_plan.strip() else original_plan


def interrupt_after_plan(memory_step, agent):
    """
    Step callback that interrupts the agent after a planning step is created.
    This allows for user interaction to review and potentially modify the plan.
    """
    if isinstance(memory_step, PlanningStep):
        print("\n🛑 Agent interrupted after plan creation...")

        # Display the created plan
        display_plan(memory_step.plan)

        # Get user choice
        choice = get_user_choice()

        if choice == 1:  # Approve plan
            print("✅ Plan approved! Continuing execution...")
            # Don't interrupt - let the agent continue
            return

        elif choice == 2:  # Modify plan
            # Get modified plan from user
            modified_plan = get_modified_plan(memory_step.plan)

            # Update the plan in the memory step
            memory_step.plan = modified_plan

            print("\nPlan updated!")
            display_plan(modified_plan)
            print("✅ Continuing with modified plan...")
            # Don't interrupt - let the agent continue with modified plan
            return

        elif choice == 3:  # Cancel
            print("❌ Execution cancelled by user.")
            agent.interrupt()
            return


def main():
    """Run the complete plan customization example"""
    print("🚀 Starting Plan Customization Example")
    print("=" * 60)

    # Create agent with planning enabled and step callback
    agent = CodeAgent(
        model=InferenceClientModel(),
        tools=[DuckDuckGoSearchTool()],  # Add a search tool for more interesting plans
        planning_interval=5,  # Plan every 5 steps for demonstration
        step_callbacks={PlanningStep: interrupt_after_plan},
        max_steps=10,
        verbosity_level=1,  # Show agent thoughts
    )

    # Define a task that will benefit from planning
    task = """Search for recent developments in artificial intelligence and provide a summary
    of the top 3 most significant breakthroughs in 2024. Include the source of each breakthrough."""

    try:
        print(f"\n📋 Task: {task}")
        print("\n🤖 Agent starting execution...")

        # First run - will create plan and potentially get interrupted
        result = agent.run(task)

        # If we get here, the plan was approved or execution completed
        print("\n✅ Task completed successfully!")
        print("\n📄 Final Result:")
        print("-" * 40)
        print(result)

    except Exception as e:
        if "interrupted" in str(e).lower():
            print("\n🛑 Agent execution was cancelled by user.")
            print("\nTo resume execution later, you could call:")
            print("agent.run(task, reset=False)  # This preserves the agent's memory")

            # Demonstrate resuming with reset=False
            print("\n" + "=" * 60)
            print("DEMONSTRATION: Resuming with reset=False")
            print("=" * 60)

            # Show current memory state
            print(f"\n📚 Current memory contains {len(agent.memory.steps)} steps:")
            for i, step in enumerate(agent.memory.steps):
                step_type = type(step).__name__
                print(f"  {i + 1}. {step_type}")

            # Ask if user wants to see resume demonstration
            resume_choice = input("\nWould you like to see resume demonstration? (y/n): ").strip().lower()
            if resume_choice == "y":
                print("\n🔄 Resuming execution...")
                try:
                    # Resume without resetting - preserves memory
                    agent.run(task, reset=False)
                    print("\n✅ Task completed after resume!")
                    print("\n📄 Final Result:")
                    print("-" * 40)
                except Exception as resume_error:
                    print(f"\n❌ Error during resume: {resume_error}")
                else:
                    print(f"\n❌ An error occurred: {e}")


if __name__ == "__main__":
    # Run the main example
    main()
