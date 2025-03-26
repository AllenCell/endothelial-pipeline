module.exports = async ({github, context, core}) => {
  // Get issue.
  const issue = await github.rest.issues.get({
    owner: context.repo.owner,
    repo: context.repo.repo,
    issue_number: context.issue.number,
  })

  // Get issue data.
  const title = issue.data.title
  const labels = issue.data.labels
  const body = issue.data.body.split("### ")

  // Separate body into individual pieces.
  const fmsid = body[1].replace("FMS ID", "").trim()
  const barcode = body[2].replace("Barcode", "").trim()
  const date = body[3].replace("Date", "").trim()
  const path = body[4].replace("Path", "").trim()
  const onenote = body[5].replace("OneNote", "").trim().split(" ")[0]
  const labkey = body[6].replace("LabKey", "").trim()
  const goal = body[7].replace("Goal", "").trim()
  const notes = body[8].replace("Notes", "").trim()
  const cell_lines = body[9].replace("Cell line(s)", "").trim()
  const flow_history = body[10].replace("Flow history", "").trim()
  const initial_flow = body[11].replace("Initial flow rate", "").trim()
  const flow_change = body[12].replace("Flow rate change", "").trim()
  const conditions_list = body[13].replace("Experimental condition(s)", "").trim()
  const microscopes = body[14].replace("Microscope(s)", "").trim()
  const objectives = body[15].replace("Objective(s)", "").trim()

  // Parse conditions checkboxes into list.
  const conditions = [...conditions_list.matchAll(/\[x\] (.+)/g)].map(e => e[1]).join(", ")

  // Format individual fields.
  let fields = [
    `**FMS ID**: ${fmsid}`,
    `**Barcode**: ${barcode}`,
    `**Date**: ${date}`,
    `**Path**: ${path}`,
    `**Cell line(s)**: ${cell_lines}`,
    `**Microscope(s)**: ${microscopes}`,
    `**Objective(s)**: ${objectives}`,
    `**Experimental condition(s)**: ${conditions}`,
    `**Goal**:\n\n${goal}`,
    `**Notes**:\n\n${notes}`,
    `**Flow settings**:\n\n_Flow history_. ${flow_history}\n\n_Initial flow_. ${initial_flow}\n\n_Flow change_. ${flow_change}`,
  ]

  // Build new issue body.
  const new_title = `${date} ${fmsid} ${barcode}`
  const new_body = `### **[🔬 LabKey](${labkey})** **[📓 OneNote](${onenote})**\n\n## ${title}\n\n${fields.join("\n\n")}`

  // Update issue labels list.
  const cell_line_labels = cell_lines.split(",").map(e => `cell line: ${e.split("(")[0].trim()}`)
  const experiment_condition_labels = conditions.split(",").map(e => `condition: ${e.trim()}`)
  const microscope_labels = microscopes.split(",").map(e => `microscope: ${e.trim()}`)
  const objective_label = objectives.split(",").map(e => `objective: ${e.trim()}`)
  const new_labels = labels.map(e => e.name).concat(cell_line_labels).concat(experiment_condition_labels).concat(microscope_labels).concat(objective_label)

  // Update issue body.
  await github.rest.issues.update({
    owner: context.repo.owner,
    repo: context.repo.repo,
    issue_number: context.issue.number,
    title: new_title,
    body: new_body,
    labels: new_labels,
  })
}
