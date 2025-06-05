

# Chat message templates with a more presentable chat interface
user_template = """
<div style="display: flex; align-items: center; margin-bottom: 10px;">
  <div style="margin-right: 8px;">👤</div>  <!-- User icon -->
  <div style="background-color: #DCF8C6; padding: 8px; border-radius: 10px; max-width: 70%;">{{message}}</div>
</div>
"""

bot_template = """
<div style="display: flex; align-items: center; margin-bottom: 10px;">
  <div style="margin-right: 8px;">🤖</div>  <!-- Bot icon -->
  <div style="background-color: #ECECEC; padding: 8px; border-radius: 10px; max-width: 70%;">{{message}}</div>
</div>
"""
