for each_class:
    p, r  = calculate_pr
    Ap_k = calculate_curve(p,r)

sum(AP) / len(classes)