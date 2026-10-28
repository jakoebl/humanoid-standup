# Repository working agreements

## Review changes before applying them

- Before editing project files, prepare a valid git `.patch` in `humanoid_standup/tmp` for the user to review 
- Always provide the clickable patch link to the patch in a normal assistant message before invoking ask_user; do not place it inside the question form.
- Apply the proposed changes only after the user explicitly approves them. Instructions such as "apply it" or "put it in" count as approval.
- When the user requests revisions to a proposal, update the patch and leave project files unchanged until approval.
- You may inspect files and test proposed changes in temporary copies without approval.
- After applying an approved patch, run the relevant checks and report the results.
- When the user asks for a terminal command, always ask whether to put it in the `COMMANDS.md` documentation in a general form.

