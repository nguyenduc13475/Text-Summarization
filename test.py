import language_tool_python

tool = language_tool_python.LanguageTool("en-US")

text = "She have a dog and it are cute."
matches = tool.check(text)
corrected = language_tool_python.utils.correct(text, matches)

print(corrected)
