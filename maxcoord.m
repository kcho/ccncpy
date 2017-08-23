function coord = maxcoord(xSPM)

[val, id] = max(xSPM.Z);
xSPM.XYZ(:,id)
