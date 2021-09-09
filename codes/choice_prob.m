function prob_A = choice_prob(V_A,V_B,b)

prob_A = exp(b*V_A) / (exp(b*V_A) + exp(b*V_B));
