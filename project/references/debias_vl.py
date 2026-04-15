import torch


def debais_vl_s_c(opts):
    if opts.dataset == 'waterbirds':
        text_descriptions = ['This is a picture of a landbird.', 'This is a picture of a waterbird.']
        spurious_prompt = ['This is a land background.', 'This is a picture of a forest.',
                           'This is a picture of a moutain.', 'This is a picture of a wood.',
                           'This is a water background.', 'This is a picture of an ocean.',
                           'This is a picture of a beach.', 'This is a picture of a port.']
        candidate_prompt = ['This is a picture of a landbird with land background.',  # 0
                            'This is a picture of a landbird with water background.',  # 1
                            'This is a picture of a landbird in the ocean',  # 2
                            'This is a picture of a landbird in the water.',  # 3
                            'This is a picture of a landbird in the forest.',  # 4
                            'This is a picture of a waterbird with land background.',  # 5
                            'This is a picture of a waterbird with water background.',  # 6
                            'This is a picture of a waterbird in the ocean',  # 7
                            'This is a picture of a waterbird in the water.',  # 8
                            'This is a picture of a waterbird in the forest.']  # 9
        S = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4],
             [5, 6], [5, 7], [5, 8], [5, 9], [6, 7], [6, 8], [6, 9], [7, 8], [7, 9], [8, 9]]
        # same bias
        B = [[0, 5], [0, 9], [4, 5], [4, 9], [5, 9], [1, 2], [1, 3], [1, 6], [1, 7], [1, 8],
             [2, 3], [2, 6], [2, 7], [2, 8], [3, 6], [3, 7], [3, 8], [6, 7], [6, 8], [7, 8]]

    if opts.dataset == 'celebA':
        text_descriptions = ['A photo of a celebrity with dark hair.', 'A photo of a celebrity with blond hair.']
        spurious_prompt = ['A photo of a male.', 'A photo of a male celebrity.', 'A photo of a man.',
                           'A photo of a female.', 'A photo of a female celebrity.', 'A photo of a woman.']

        candidate_prompt = ['A photo of a male celebrity with dark hair.',
                            'A photo of a female celebrity with dark hair.',
                            'A photo of a male celebrity with blond hair.',
                            'A photo of a female celebrity with blond hair.']
        S = [[0, 1], [2, 3]]
        B = [[0, 2], [1, 3]]  # same bias
    
    if opts.dataset == 'counteranimal':
        # Generic setup for counteranimal - text descriptions will be set dynamically
        text_descriptions = []  # Will be populated from dataset

        # Spurious backgrounds: common vs counter
        spurious_prompt = ['A photo with common background.', 'A photo with natural habitat.',
                           'A photo with typical background.', 'A photo with counter background.',
                           'A photo with unusual background.', 'A photo with atypical background.']

        # Simple candidate prompt structure (will work for zero-shot)
        candidate_prompt = ['A photo of an animal with common background.',
                            'A photo of an animal with counter background.']

        # Minimal S and B matrices for basic setup
        S = [[0, 1]]
        B = [[0, 1]]

    if opts.dataset == 'fairface':
        # Stereotype-based debiasing for FairFace (BendVLM method)
        # Goal: Remove gender bias from stereotype queries (25 queries)
        text_descriptions = [
            'A photo of an addict.',
            'A photo of a burglar.',
            'A photo of a criminal.',
            'A photo of a dangerous person.',
            'A photo of a delinquent.',
            'A photo of a deviant.',
            'A photo of an embezzler.',
            'A photo of a felon.',
            'A photo of a fraud.',
            'A photo of a freak.',
            'A photo of a gangster.',
            'A photo of a hacker.',
            'A photo of an illegal person.',
            'A photo of an insane person.',
            'A photo of a lawless person.',
            'A photo of a mugger.',
            'A photo of a murderer.',
            'A photo of a psychopath.',
            'A photo of a robber.',
            'A photo of a savage.',
            'A photo of a shoplifter.',
            'A photo of a terrorist.',
            'A photo of a thief.',
            'A photo of a thug.',
            'A photo of a violent person.',
        ]

        # Spurious prompts for gender (confounder to remove)
        spurious_prompt = [
            'A photo of a male.', 'A photo of a man.', 'A photo of a male person.',
            'A photo of a female.', 'A photo of a woman.', 'A photo of a female person.'
        ]

        # Candidate prompts: stereotype x gender combinations (25 stereotypes x 2 genders = 50)
        candidate_prompt = [
            'A photo of a male addict.',           'A photo of a female addict.',           # 0, 1
            'A photo of a male burglar.',          'A photo of a female burglar.',          # 2, 3
            'A photo of a male criminal.',         'A photo of a female criminal.',         # 4, 5
            'A photo of a dangerous male person.', 'A photo of a dangerous female person.', # 6, 7
            'A photo of a male delinquent.',       'A photo of a female delinquent.',       # 8, 9
            'A photo of a male deviant.',          'A photo of a female deviant.',          # 10, 11
            'A photo of a male embezzler.',        'A photo of a female embezzler.',        # 12, 13
            'A photo of a male felon.',            'A photo of a female felon.',            # 14, 15
            'A photo of a male fraud.',            'A photo of a female fraud.',            # 16, 17
            'A photo of a male freak.',            'A photo of a female freak.',            # 18, 19
            'A photo of a male gangster.',         'A photo of a female gangster.',         # 20, 21
            'A photo of a male hacker.',           'A photo of a female hacker.',           # 22, 23
            'A photo of a male illegal person.',   'A photo of a female illegal person.',   # 24, 25
            'A photo of a male insane person.',    'A photo of a female insane person.',    # 26, 27
            'A photo of a male lawless person.',   'A photo of a female lawless person.',   # 28, 29
            'A photo of a male mugger.',           'A photo of a female mugger.',           # 30, 31
            'A photo of a male murderer.',         'A photo of a female murderer.',         # 32, 33
            'A photo of a male psychopath.',       'A photo of a female psychopath.',       # 34, 35
            'A photo of a male robber.',           'A photo of a female robber.',           # 36, 37
            'A photo of a male savage.',           'A photo of a female savage.',           # 38, 39
            'A photo of a male shoplifter.',       'A photo of a female shoplifter.',       # 40, 41
            'A photo of a male terrorist.',        'A photo of a female terrorist.',        # 42, 43
            'A photo of a male thief.',            'A photo of a female thief.',            # 44, 45
            'A photo of a male thug.',             'A photo of a female thug.',             # 46, 47
            'A photo of a violent male person.',   'A photo of a violent female person.',   # 48, 49
        ]

        # S: same stereotype, different gender (semantic preservation)
        # [[0,1], [2,3], [4,5], ...] for all 25 stereotypes
        S = [[i*2, i*2+1] for i in range(25)]

        # B: same gender, different stereotype (bias direction)
        # All pairs of males (even indices) and all pairs of females (odd indices)
        B = []
        male_indices = list(range(0, 50, 2))    # 0, 2, 4, ..., 48
        female_indices = list(range(1, 50, 2))  # 1, 3, 5, ..., 49
        for i in range(len(male_indices)):
            for j in range(i+1, len(male_indices)):
                B.append([male_indices[i], male_indices[j]])
        for i in range(len(female_indices)):
            for j in range(i+1, len(female_indices)):
                B.append([female_indices[i], female_indices[j]])

    return spurious_prompt, candidate_prompt, S, B, text_descriptions


def debias_vl(spurious_embeddings, candidate_embeddings, S):
    P0 = get_proj_matrix(spurious_embeddings)  # 4.2 remove pseudo way
    M = get_M(candidate_embeddings, S)  # S is opposite index in candidate -> M = zizj -zjzi
    # Regularization ensures that the eigenvalues are bounded away from zero, thus making the matrix invertible and the system stable.
    G = 1000 * M + torch.eye(M.shape[0])
    P = P0 @ torch.inverse(G)
    return P


def bias_vl(spurious_embeddings, candidate_embeddings, B, lamda=1000):
    P0, proj_sup = get_proj_matrix(spurious_embeddings, bias='bias')
    shape_ = proj_sup.shape[0]
    P_set = []
    for b in B:
        M = get_A(candidate_embeddings[b[0]], candidate_embeddings[b[1]])
        # Regularization ensures that the eigenvalues are bounded away from zero, thus making the matrix invertible and the system stable.
        G = lamda * M + torch.eye(shape_)
        P = P0 @ torch.inverse(G)
        P_set.append(P)
    return torch.stack(P_set)


def get_proj_matrix(embeddings, bias=None):
    # Perform SVD
    U, S, V = torch.svd(embeddings)

    # Use all components
    basis = V

    # Orthogonal projection
    proj = torch.inverse(basis.T @ basis)
    proj = basis @ proj
    proj_sup = proj @ basis.T
    proj = torch.eye(proj.shape[0]) - proj_sup
    if bias is not None:
        return proj, proj_sup
    return proj


def get_A(z_i, z_j):
    z_i = z_i[:, None]
    z_j = z_j[:, None]
    return z_i @ z_i.T + z_j @ z_j.T - z_i @ z_j.T - z_j @ z_i.T


def get_M(embeddings, S):
    d = embeddings.shape[1]
    M = torch.zeros((d, d), device=embeddings.device)
    for s in S:
        M += get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(S)
