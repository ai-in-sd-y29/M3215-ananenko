class ToolError(Exception):
    """Ожидаемая ошибка инструмента: код для программы и сообщение без секретов."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
