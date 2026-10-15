from aiogram import Router

from app.handlers import agent_commands, chat, commands, reminders

# Команды и кнопки регистрируются раньше общего текстового обработчика.
router = Router(name="assistant")
router.include_router(commands.router)
router.include_router(agent_commands.router)
router.include_router(reminders.router)
router.include_router(chat.router)
