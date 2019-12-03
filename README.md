# sandBarTool
develop morphology metrics related to sandbar evolution, formation, and modeling.  At this point the logic is below

To identify sandbars:
1. takes a mean profile trend (given), subtracts it (leaving orange line).
2. then finds peaks on that line (inital bar locations, in small red circles).  
3. It then moves back to the profile, finds the nearest shoreward point of inflection and marks that
for each bar location
- the bar has to have a point of inflection (slope <0, positive right to left) to be identified as a bar

To identify Troughs:
 - it finds the minimum elevation shoreward of each bar location previously identified.
 - The trough has to be less than the bar elevation, thereby not always identifying a trough if a terrace is present
 
 plans: 
  - take the 2D grid with ID'd sandbars and troughs, run an alongshore filter to remove cross-shore noise
  
