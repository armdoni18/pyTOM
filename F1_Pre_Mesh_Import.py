import numpy as np

def F1_Pre_Mesh_Import(inputs):
    if isinstance(inputs, str):
        modelname = inputs
    elif isinstance(inputs, dict):
        modelname = inputs.get("modelname", None)
        if modelname is None:
            raise KeyError('inputs dict must contain key "modelname"')
    else:
        raise TypeError("inputs must be str or dict")

    mshfile = modelname + ".msh"

    msh = {}
    surf2dom = {}

    flag_category = 0
    list_category = (
        "$MeshFormat", "$EndMeshFormat",
        "$PhysicalNames", "$EndPhysicalNames",
        "$Entities", "$EndEntities",
        "$Nodes", "$EndNodes",
        "$Elements", "$EndElements"
    )
    flag_state = 0

    with open(mshfile, "r") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]

    for line in lines:
        if line and line[0] == "$":
            for i in range(len(list_category)):
                if line == list_category[i]:
                    flag_category = i
                    flag_state = 0
                    continue

        else:   ## Data parsing
            if flag_category == 2:
                if flag_state == 0:
                    if len(line) == 1:
                        num_phys_entities = int(line)
                        flag_state = 1

                        phys_tag_str = {
                            "Air": [], "Design": [], "NonDesign": [],
                            "Coil1": [], "Coil2": [], "FixIron": [],
                            "PM1": [], "PM2": []
                        }
                        ind_phys_entities = 0

                elif flag_state == 1:
                    X = line.split()
                    tag = int(X[1])
                    name = X[2]

                    if name.startswith('"Air"'):
                        phys_tag_str["Air"].append(tag)
                    elif name.startswith('"Design"'):
                        phys_tag_str["Design"].append(tag)
                    elif name.startswith('"Coil1"'):
                        phys_tag_str["Coil1"].append(tag)
                    elif name.startswith('"Coil2"'):
                        phys_tag_str["Coil2"].append(tag)
                    elif name.startswith('"NonDesign"'):
                        phys_tag_str["NonDesign"].append(tag)
                    elif name.startswith('"FixIron"'):
                        phys_tag_str["FixIron"].append(tag)
                    elif name.startswith('"PM1"'):
                        phys_tag_str["PM1"].append(tag)
                    elif name.startswith('"PM2"'):
                        phys_tag_str["PM2"].append(tag)

                    ind_phys_entities += 1
                    if ind_phys_entities >= num_phys_entities:
                        flag_state = 0

            # ========= Entities =========
            elif flag_category == 4:
                if flag_state == 0:
                    num_point, num_line, num_surf, num_vol = [int(i) for i in line.split()]

                    Air_list = []
                    Design_list = []
                    NonDesign_list = []
                    Coil1_list = []
                    Coil2_list = []
                    FixIron_list = []
                    PM1_list = []
                    PM2_list = []

                    ent_stage = "point"
                    ent_i = 0
                    flag_state = 1

                elif flag_state == 1:
                    while True:
                        if ent_stage == "point" and num_point == 0:
                            ent_stage = "line"; continue
                        if ent_stage == "line" and num_line == 0:
                            ent_stage = "surf"; continue
                        if ent_stage == "surf" and num_surf == 0:
                            ent_stage = "vol"; continue
                        if ent_stage == "vol" and num_vol == 0:
                            ent_stage = "done"; continue
                        break

                    if ent_stage == "point":
                        ent_i += 1
                        if ent_i >= num_point:
                            ent_stage = "line"; ent_i = 0

                    elif ent_stage == "line":
                        ent_i += 1
                        if ent_i >= num_line:
                            ent_stage = "surf"; ent_i = 0

                    elif ent_stage == "surf":
                        ent_i += 1

                        toks = line.split()
                        surf_id = int(toks[0])

                        nphys = int(toks[7])
                        if nphys > 0:
                            phys_tags = list(map(int, toks[8:8 + nphys]))
                            phys = phys_tags[0]

                            if phys in phys_tag_str["Air"]:
                                Air_list.append(surf_id)
                            elif phys in phys_tag_str["Design"]:
                                Design_list.append(surf_id)
                            elif phys in phys_tag_str["Coil1"]:
                                Coil1_list.append(surf_id)
                            elif phys in phys_tag_str["Coil2"]:
                                Coil2_list.append(surf_id)
                            elif phys in phys_tag_str["FixIron"]:
                                FixIron_list.append(surf_id)
                            elif phys in phys_tag_str["NonDesign"]:
                                NonDesign_list.append(surf_id)
                            elif phys in phys_tag_str["PM1"]:
                                PM1_list.append(surf_id)
                            elif phys in phys_tag_str["PM2"]:
                                PM2_list.append(surf_id)

                        if ent_i >= num_surf:
                            surf2dom.clear()
                            for s in Air_list:      surf2dom[s] = 1
                            for s in Design_list:   surf2dom[s] = 2
                            for s in Coil1_list:    surf2dom[s] = 3
                            for s in Coil2_list:    surf2dom[s] = 4
                            for s in NonDesign_list:surf2dom[s] = 5
                            for s in FixIron_list:    surf2dom[s] = 6
                            for s in PM1_list:      surf2dom[s] = 7
                            for s in PM2_list:      surf2dom[s] = 8

                            ent_stage = "vol"
                            ent_i = 0

                    elif ent_stage == "vol":
                        ent_i += 1
                        if ent_i >= num_vol:
                            ent_stage = "done"

            # ========= Nodes =========
            elif flag_category == 6:
                if flag_state == 0:
                    X = [int(i) for i in line.split()]
                    num_node = X[1]
                    p = np.zeros((num_node, 2))
                    flag_state = 1

                elif flag_state == 1:
                    X = [int(i) for i in line.split()]
                    num_loop = X[3]
                    if num_loop > 0:
                        ind_list = np.zeros(num_loop, dtype=int)
                        ind_inner_loop = 0
                        flag_state = 2

                elif flag_state == 2:
                    # keep internal index 0-based for coordinate storage
                    ind_list[ind_inner_loop] = int(line) - 1
                    ind_inner_loop += 1
                    if ind_inner_loop >= len(ind_list):
                        ind_inner_loop = 0
                        flag_state = 3

                elif flag_state == 3:
                    X = [float(i) for i in line.split()]
                    p[ind_list[ind_inner_loop], :] = X[0:2]
                    ind_inner_loop += 1
                    if ind_inner_loop >= len(ind_list):
                        flag_state = 1
                        ind_inner_loop = 0

            # ========= Elements =========
            elif flag_category == 8:
                if flag_state == 0:
                    X = [int(i) for i in line.split()]
                    num_blocks = X[0]
                    t_list = []
                    flag_state = 1

                elif flag_state == 1:
                    X = [int(i) for i in line.split()]
                    dim, surf_tag, etype, num_elem = X
                    ind_inner_loop = 0

                    if dim == 2 and etype == 2:
                        flag_condition = surf2dom.get(surf_tag, 1)
                        flag_state = 2
                    else:
                        flag_state = 3

                elif flag_state == 2:
                    X = [int(i) for i in line.split()]
                    n1, n2, n3 = X[1:4]

                    # ✅ IMPORTANT CHANGE: keep connectivity 1-based (like structural)
                    t_list.append([n1, n2, n3, flag_condition])

                    ind_inner_loop += 1
                    if ind_inner_loop >= num_elem:
                        flag_state = 1

                elif flag_state == 3:
                    ind_inner_loop += 1
                    if ind_inner_loop >= num_elem:
                        flag_state = 1

    msh["X"] = p
    msh["IX"] = np.array(t_list, dtype=int)
    msh["ne"] = msh["IX"].shape[0]
    return msh
