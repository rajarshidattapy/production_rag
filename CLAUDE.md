# Project instructions

## Keep `docs/feature.md` up to date

`docs/feature.md` lists what still has to be fixed for production and for scale. Whenever your work on this repo reveals something that belongs there, add it in the same session, without being asked. This applies to every agent and subagent, whatever the task was.

**Add an item when** you find a real problem in the current code that would:

- break, slow down, or get expensive as users, documents, or servers grow
- make it unsafe or unreliable to run in production

**Don't add** polish, style, diagram or documentation notes, feature ideas, or limitations people can easily live with. Keep the list short. Don't go hunting for items unless the task asks for it: record what you come across while doing your work.

**How to add one:**

1. Read `docs/feature.md` first. If the problem is already listed, update that item with anything new instead of adding a second one.
2. Put it in the section that fits:
   - **Before going to production:** needed before the first real deployment.
   - **Fix when scaling:** under the group that fits, roughly in the order things will hurt.
3. Match the format of the existing items: a bold one-line statement of the problem, why it matters, and the fix if it's known.
4. For items under **Fix when scaling**, end with how it was confirmed: *Reproduced*, *Measured* (with the numbers), or *From the code*. Only claim what you actually did.

**Remove an item** in the same change that fixes it. If you fix only part of it, reword it to say what's left.

Only edit `docs/feature.md` for this. Don't fix the problem as well unless your task covers it, and leave committing to the user. When you finish, tell the user what you added or removed.
