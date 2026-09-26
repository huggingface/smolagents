# docstyle-ignore
INSTALL_CONTENT = """
# Installation
! pip install smolagents
# Pour installer depuis la source au lieu de la dernière version, commentez la commande ci-dessus et décommentez la suivante.
# ! pip install git+https://github.com/huggingface/smolagents.git
"""

notebook_first_cells = [{"type": "code", "content": INSTALL_CONTENT}]
black_avoid_patterns = {
    "{processor_class}": "FakeProcessorClass",
    "{model_class}": "FakeModelClass",
    "{object_class}": "FakeObjectClass",
}
