import os
import logging
from typing import Optional, List
from smolagents import Tool

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MemoryBankTool(Tool):
    name = "memory_bank"
    description = """
    Tool for interacting with the agent's Memory Bank, a structured collection of markdown files
    that maintain continuity and coherence across problem-solving steps. The Memory Bank includes
    sections for Project Overview, Contextual Information, Technical Specifications,
    Active Progress Tracking, and Historical Progress.

    Before each step, the agent should consult the Memory Bank to align actions with project objectives and avoid redundancy.
    After each step, the agent should update the Memory Bank to reflect changes, insights gained, and decisions made.

    The Memory Bank is stored in {memory_bank_dir_path}
    """
    inputs = {
        "action": {
            "type": "string",
            "description": "The action to perform: 'read', 'update', 'list', or 'create'."
        },
        "section": {
            "type": "string",
            "description": "The Memory Bank section to interact with: 'Project Overview', 'Contextual Information', 'Technical Specifications', "
                           "'Active Progress Tracking', or 'Historical Progress'."
                           "Required for 'read', 'update', and 'create' actions.",
            "nullable": True,
        },
        "content": {
            "type": "string",
            "description": "The content to write when using the 'update' or 'create' action. "
                           "Required for 'update' and 'create' actions.",
            "nullable": True,
        },
    }
    output_type = "string"

    # Default Memory Bank sections
    DEFAULT_SECTIONS = [
        "Project Overview",
        "Contextual Information",
        "Technical Specifications",
        "Active Progress Tracking",
        "Historical Progress"
    ]

    def __init__(self, memory_bank_dir_path: str):
        """
        Initialize the MemoryBankTool.

        Args:
            memory_bank_dir_path: Path to the directory where Memory Bank files are stored.
        """
        super().__init__()
        self.memory_bank_dir_path = os.path.abspath(memory_bank_dir_path)

        # Update description with the actual memory bank directory path
        self.description = self.description.format(memory_bank_dir_path=str(self.memory_bank_dir_path))

        # Ensure the memory bank directory exists
        os.makedirs(self.memory_bank_dir_path, exist_ok=True)

        # Initialize default sections if they don't exist
        self._initialize_default_sections()

        logger.info(f"MemoryBankTool initialized with directory: {self.memory_bank_dir_path}")

    def _initialize_default_sections(self) -> None:
        """Initialize default Memory Bank sections if they don't exist."""
        for section in self.DEFAULT_SECTIONS:
            section_path = self._get_section_path(section)
            if not os.path.exists(section_path):
                with open(section_path, "w", encoding="utf-8") as f:
                    f.write(f"# {section}\n\n")
                logger.info(f"Created default Memory Bank section: {section}")

    def _validate_path(self, path: str) -> str:
        """
        Validate that a path is within the Memory Bank directory.

        Args:
            path: Path to validate.

        Returns:
            Absolute path if valid.

        Raises:
            ValueError: If path is invalid or outside the Memory Bank directory.
        """
        if not path:
            raise ValueError("Path cannot be empty")

        # Convert to absolute path
        abs_path = os.path.abspath(path)

        # Basic path validation
        if not os.path.normpath(abs_path):
            raise ValueError(f"Invalid path: {path}")

        # Validate that path is within memory bank directory
        if not abs_path.startswith(self.memory_bank_dir_path):
            raise ValueError(
                f"Path {abs_path} is outside of the Memory Bank directory {self.memory_bank_dir_path}"
            )

        return abs_path

    def _get_section_path(self, section: str) -> str:
        """
        Get the file path for a Memory Bank section.

        Args:
            section: Name of the section.

        Returns:
            Path to the section file.
        """
        # Sanitize section name for use as filename
        filename = f"{section.replace(' ', '_').lower()}.md"
        return os.path.join(self.memory_bank_dir_path, filename)

    def _validate_section(self, section: str) -> str:
        """
        Validate a Memory Bank section name and return its file path.

        Args:
            section: Name of the section.

        Returns:
            Path to the section file if valid.

        Raises:
            ValueError: If section is invalid.
        """
        if not section:
            raise ValueError("Section name cannot be empty")

        section_path = self._get_section_path(section)
        return self._validate_path(section_path)

    def _list_sections(self) -> List[str]:
        """
        List all available Memory Bank sections.

        Returns:
            List of section names.
        """
        sections = []
        for file in os.listdir(self.memory_bank_dir_path):
            if file.endswith(".md"):
                # Convert filename back to section name
                section = file[:-3].replace("_", " ").title()
                sections.append(section)
        return sorted(sections)

    def _read_section(self, section: str) -> str:
        """
        Read the content of a Memory Bank section.

        Args:
            section: Name of the section.

        Returns:
            Content of the section.

        Raises:
            FileNotFoundError: If section file doesn't exist.
        """
        section_path = self._validate_section(section)

        if not os.path.exists(section_path):
            raise FileNotFoundError(f"Memory Bank section not found: {section}")

        with open(section_path, "r", encoding="utf-8") as f:
            content = f.read()

        return content

    def _update_section(self, section: str, content: str) -> str:
        """
        Update the content of a Memory Bank section.

        Args:
            section: Name of the section.
            content: New content for the section.

        Returns:
            Success message.

        Raises:
            FileNotFoundError: If section file doesn't exist.
        """
        section_path = self._validate_section(section)

        if not os.path.exists(section_path):
            raise FileNotFoundError(f"Memory Bank section not found: {section}")

        with open(section_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully updated Memory Bank section: {section}"

    def _create_section(self, section: str, content: str) -> str:
        """
        Create a new Memory Bank section.

        Args:
            section: Name of the new section.
            content: Content for the new section.

        Returns:
            Success message.
        """
        section_path = self._get_section_path(section)

        # Validate that the path is within the Memory Bank directory
        self._validate_path(section_path)

        # Check if section already exists
        if os.path.exists(section_path):
            return f"Memory Bank section already exists: {section}. Use 'update' action to modify it."

        # Create the section file
        with open(section_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully created Memory Bank section: {section}"

    def forward(
        self,
        action: str,
        section: Optional[str] = None,
        content: Optional[str] = None,
    ) -> str:
        """
        Execute the Memory Bank tool with the specified action.

        Args:
            action: The action to perform: 'read', 'update', 'list', or 'create'.
            section: The Memory Bank section to interact with.
            content: The content to write when using the 'update' or 'create' action.

        Returns:
            Result of the action.
        """
        try:
            # Validate action
            action = action.lower()
            if action not in ["read", "update", "list", "create"]:
                raise ValueError(
                    f"Invalid action: {action}. Must be one of: 'read', 'update', 'list', 'create'."
                )

            # Execute action
            if action == "list":
                sections = self._list_sections()
                return "Available Memory Bank sections:\n" + "\n".join(f"- {s}" for s in sections)

            # Validate section for other actions
            if not section:
                raise ValueError(f"Section name is required for '{action}' action.")

            if action == "read":
                return self._read_section(section)

            # Validate content for update and create actions
            if action in ["update", "create"] and not content:
                raise ValueError(f"Content is required for '{action}' action.")

            if action == "update" and content is not None:
                return self._update_section(section, content)

            if action == "create" and content is not None:
                return self._create_section(section, content)

            # If we reach here without returning, something went wrong
            return "Invalid action or missing required parameters"

        except Exception as e:
            error_msg = f"Memory Bank tool failed: {str(e)}"
            logger.error(error_msg)
            return error_msg
