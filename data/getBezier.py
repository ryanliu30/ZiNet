# Convert SVG to Bezier Curves 
# from_svg_path(path_str, shape_to_canvas = torch.eye(3), force_close = False):
# Code snippet from https://stackoverflow.com/questions/59059991/svg-path-not-parsing-correctly-svgpathtools-inkscape
# Input:  directory path for an SVG file 
# Output: a list of diffsvg paths which are all cubic bezier curves and are the original curves cut into 8 pieces. (8 is a hyperparam) 

def getBeziers(svg_path, pieces = 8):
    mydoc = minidom.parse(svg_path)
    path_tag = mydoc.getElementsByTagName("path")
    d_string = path_tag[0].attributes['d'].value
    x = from_svg_path(d_string) 
    for curve in x:
        paths = []
        points = curve.points.clone()
        num_control_points = curve.num_control_points.clone()
        points = torch.cat([points, points[0].unsqueeze(0)], dim = 0) # push element 0 to last point
        anchors = np.cumsum(num_control_points.clone() + 1) 
        anchors = torch.cat([torch.tensor([0]), anchors], dim = 0)
        # anchors[i] and anchors[i + 1] are indices of anchors, anything in between are control points
        for idx in range(len(anchors) - 1):
            bezierCurve = bezier.Curve(points[anchors[idx] : anchors[idx + 1] + 1, :].T, degree = anchors[idx + 1] - anchors[idx])
            while bezierCurve.degree < 3: # Make sure everything is cubic
                bezierCurve = bezierCurve.elevate()
            # Split curve into 8 parts 
            parts = []
            for i in range(pieces):
                parts.append(bezierCurve.specialize(i / 8, (i + 1) / 8))
            paths = np.concatenate([paths, parts], axis = 0)
        
        newpoints = np.empty((0, 2))
        new_controls = np.empty(0)
        
        for path in paths: # get paths back into diffsvg format
            newpoints = np.concatenate([newpoints, path.nodes.T[ : -1, :]], axis = 0)
            new_controls = np.concatenate([new_controls, [path.degree - 1]])
        
        curve.points = torch.tensor(newpoints)
        curve.num_control_points = torch.tensor(new_controls)
        #print(newpoints)
        #print(new_controls)
    #for c in curves:
    #    c.plot(10, ax = plt)
    print(len(x[0].points))
    
    print(len(x[0].num_control_points))
    return x
    