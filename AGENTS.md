# Repository working agreements

## Review changes before applying them

- Before editing project files, prepare a `.patch` in `/tmp` for the user to review and provide its path.
- Apply the proposed changes only after the user explicitly approves them. Instructions such as "apply it" or "put it in" count as approval.
- When the user requests revisions to a proposal, update the patch and leave project files unchanged until approval.
- You may inspect files and test proposed changes in temporary copies without approval.
- After applying an approved patch, run the relevant checks and report the results.
- An explicit instruction to edit a particular file directly takes precedence over this default workflow.
