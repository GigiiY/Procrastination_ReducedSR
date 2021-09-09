clear all
% set the seed of random number
rand_seed = 210626;
rand('twister',rand_seed);

% parameters
n = 20; % number of episodes (trials)
m = 10000;
num_state = 5; % number of states
a_modes = [1:3]; % mode of learning rate, 1:constant at 0.2, 2:constant at 0.4, 3:variable (original)
b_set = [10 20 30];
g_set = [0.85 0.95]; % time discount factor
for k_a = 1:length(a_modes)
    for k_b = 1:length(b_set)
        b = b_set(k_b);
        for k_g = 1:length(g_set)
            g = g_set(k_g);
            x = [g^4 g^3 g^2 g 1]; % feature variables for [S1 S2 S3 S4 S5]
            
            % cost and reward
            cost_set = [0.05 0.1];
            for k_cost = 1:length(cost_set)
                cost = cost_set(k_cost);
                costs = cost*ones(1,num_state-1);
                rewards = [-costs 1];
                
                % estimated true value of GO and STAY
                v_GO{k_a}{k_b}{k_g}{k_cost} = NaN(n,num_state-1,m);
                v_STAY{k_a}{k_b}{k_g}{k_cost} = NaN(n,num_state-1,m);
                
                % approximated value of GO and STAY
                v_GO2{k_a}{k_b}{k_g}{k_cost} = NaN(n,num_state-1,m);
                v_STAY2{k_a}{k_b}{k_g}{k_cost} = NaN(n,num_state-1,m);
                
                % estimated true state value
                state_value{k_a}{k_b}{k_g}{k_cost} = NaN(n,num_state,m);
                
                % weight
                w_history{k_a}{k_b}{k_g}{k_cost} = NaN(n,num_state);
                w_history2{k_a}{k_b}{k_g}{k_cost} = NaN(n,num_state,m);
                
                % number of STAY
                num_STAY{k_a}{k_b}{k_g}{k_cost} = zeros(n,num_state-1,m);
                
                % School term (episodes without procrastination)
                w = 0;
                for k_episode = 1:n
                    if k_a == 1
                        a = 0.2;
                    elseif k_a == 2
                        a = 0.4;
                    elseif k_a == 3
                        a = (0.5/(1+0.2*k_episode)); % learning rate
                    end
                    
                    for k_state = 1:num_state-1
                        TDerror = rewards(k_state) + g*w*x(k_state+1) - w*x(k_state);
                        w = w + a*TDerror*x(k_state);
                        w_history{k_a}{k_b}{k_g}{k_cost}(k_episode,k_state) = w;
                    end
                    % at the last (goal) state
                    k_state = num_state;
                    TDerror = rewards(k_state) + 0 - w*x(k_state); % because there is no "next state"
                    w = w + a*TDerror*x(k_state);
                    w_history{k_a}{k_b}{k_g}{k_cost}(k_episode,k_state) = w;
                end
                
                % true state value after School term / before Vacation period
                c = zeros(1,num_state-1);
                state_value_initial = NaN(1,num_state);
                for k1 = 1:num_state-1
                    for k2 = k1:num_state-1
                        c(k1)=c(k1)+g^(k2-k1)*costs(k2);
                    end
                    state_value_initial(k1)=g^(num_state-k1)-c(k1);
                end
                k1 = num_state;
                state_value_initial(k1) = 1;
                
                % Vacation period (episodes with procrastination)
                for k = 1:m
                    fprintf('%d-%d-%d-%d-%d\n',k_a,k_b,k_g,k_cost,k);
                    w_2 = w;
                    
                    state_value_current = state_value_initial;
                    
                    for k_episode = 1:n
                        if k_a == 1
                            a = 0.2;
                        elseif k_a == 2
                            a = 0.4;
                        elseif k_a == 3
                            a = (0.5/(1+0.2*k_episode)); % learning rate
                        end
                        
                        for k_state = 1:num_state-1
                            Vest_STAY = g*w_2*x(k_state);
                            Vest_GO = g*w_2*x(k_state+1) - costs(k_state);
                            Prob_STAY = choice_prob(Vest_STAY,Vest_GO,b);
                            
                            v_GO2{k_a}{k_b}{k_g}{k_cost}(k_episode,k_state,k) = Vest_GO;
                            v_STAY2{k_a}{k_b}{k_g}{k_cost}(k_episode,k_state,k) = Vest_STAY;
                            v_GO{k_a}{k_b}{k_g}{k_cost}(k_episode,k_state,k) = g*state_value_current(k_state+1) - costs(k_state);
                            v_STAY{k_a}{k_b}{k_g}{k_cost}(k_episode,k_state,k) = g*state_value_current(k_state);
                            
                            %codes for STAY
                            while rand < Prob_STAY % choose STAY with probability of Prob_STAY
                                TDerror = 0 + g*w_2*x(k_state) - w_2*x(k_state);
                                w_2 = w_2 + a*TDerror*x(k_state);
                                num_STAY{k_a}{k_b}{k_g}{k_cost}(k_episode,k_state,k) = num_STAY{k_a}{k_b}{k_g}{k_cost}(k_episode,k_state,k)+1;
                                
                                TDerror_true = 0 + g*state_value_current(k_state) - state_value_current(k_state);
                                state_value_current(k_state) = state_value_current(k_state) + a*TDerror_true;
                                
                                Vest_STAY = g*w_2*x(k_state);
                                Vest_GO = g*w_2*x(k_state+1) - costs(k_state);
                                Prob_STAY = choice_prob(Vest_STAY,Vest_GO,b);
                            end
                            
                            %codes for GO
                            TDerror = rewards(k_state) + g*w_2*x(k_state+1) - w_2*x(k_state);
                            w_2 = w_2 + a*TDerror*x(k_state);
                            
                            TDerror_true = rewards(k_state) + g*state_value_current(k_state+1) - state_value_current(k_state);
                            state_value_current(k_state) = state_value_current(k_state) + a*TDerror_true;
                            w_history2{k_a}{k_b}{k_g}{k_cost}(k_episode,k_state,k) = w_2;
                        end
                        
                        % at the last (goal) state
                        k_state = num_state;
                        TDerror = rewards(k_state) + 0 - w_2*x(k_state); % because there is no "next state"
                        w_2 = w_2 + a*TDerror*x(k_state);
                        w_history2{k_a}{k_b}{k_g}{k_cost}(k_episode,k_state,k) = w_2;
                        
                        state_value{k_a}{k_b}{k_g}{k_cost}(k_episode,:,k) = state_value_current;
                    end
                end
            end
        end
    end
end

% integrate and save the data
data.v_GO = v_GO;
data.v_STAY = v_STAY;
data.v_GO2 = v_GO2;
data.v_STAY2 = v_STAY2;
data.state_value = state_value;
data.w_history = w_history;
data.w_history2 = w_history2;
data.num_STAY = num_STAY;
save procra_data2 data

% make figures
save_fig = 1; % 1:save figures, 0:not save figures

%
load procra_data2
n = 20; % number of episodes (trials)
num_state = 5; % number of states
m = 10000;
a_modes = [1:3];
b_set = [10 20 30];
g_set = [0.85 0.95];
cost_set = [0.05 0.1];

% case with a specified set of parameters
k_a = 2; % original: 3
k_b = 2; % original: 2
k_g = 1; % original: 1
k_cost = 2; % original: 2
g = g_set(k_g); % time discount factor
x = [g^4 g^3 g^2 g 1]; % feature variables for [S1 S2 S3 S4 S5]
mean_num_STAY = mean(data.num_STAY{k_a}{k_b}{k_g}{k_cost},3);
std_num_STAY = std(data.num_STAY{k_a}{k_b}{k_g}{k_cost},0,3);
d_est = data.v_GO2{k_a}{k_b}{k_g}{k_cost} - data.v_STAY2{k_a}{k_b}{k_g}{k_cost};
mean_d_est = mean(d_est,3);
std_d_est = std(d_est,0,3);
d_true = data.v_GO{k_a}{k_b}{k_g}{k_cost} - data.v_STAY{k_a}{k_b}{k_g}{k_cost};
mean_d_true = mean(d_true,3);
std_d_true = std(d_true,0,3);

% line graph for w (Fig 8C)
tmp_data{1} = data.w_history{k_a}{k_b}{k_g}{k_cost};
tmp_data{2} = mean(data.w_history2{k_a}{k_b}{k_g}{k_cost},3);
tmp_title = {'School Term','Vacation'};
for k1 = 1:2
    F = figure;
    A = axes;
    hold on;
    axis([0 21 -0.2 1]);
    P = plot([0 21],[0 0],'k:');
    for k2 = 1:size(tmp_data{k1},2)
        P = plot(tmp_data{k1}(:,k2)); set(P,'Color',(k2*0.16)*[1 1 1],'LineWidth',2);
    end
    title(tmp_title{k1},'FontSize',24);
    xlabel('Episode','FontSize',24);
    ylabel({'Coefficient w of','Approximated Value Function'},'FontSize',24);
    set(A,'XTick',[5:5:20],'XTickLabel',[5:5:20]);
    set(A,'YTick',[-0.2:0.2:1],'YTickLabel',[-0.2:0.2:1]);
    set(A,'FontSize',20);
    if save_fig
        print(F,'-depsc',['Fig8C' num2str(k1)]);
    end
end

% varying parameters
k_a_set = [3 3 1 2 3 3];
k_b_set = [2 2 2 2 1 3];
k_g_set = [1 2 1 1 1 1];
k_cost_set = [1 1 2 2 2 2];
tmp_fignamge_set = {'7B','7C','8A','8B','8D','8E'};
for k_para = 1:6
    
    % case with a specified set of parameters
    k_a = k_a_set(k_para); % original: 3
    k_b = k_b_set(k_para); % original: 2
    k_g = k_g_set(k_para); % original: 1
    k_cost = k_cost_set(k_para); % original: 2
    g = g_set(k_g); % time discount factor
    x = [g^4 g^3 g^2 g 1]; % feature variables for [S1 S2 S3 S4 S5]
    mean_num_STAY = mean(data.num_STAY{k_a}{k_b}{k_g}{k_cost},3);
    std_num_STAY = std(data.num_STAY{k_a}{k_b}{k_g}{k_cost},0,3);
    d_est = data.v_GO2{k_a}{k_b}{k_g}{k_cost} - data.v_STAY2{k_a}{k_b}{k_g}{k_cost};
    mean_d_est = mean(d_est,3);
    std_d_est = std(d_est,0,3);
    d_true = data.v_GO{k_a}{k_b}{k_g}{k_cost} - data.v_STAY{k_a}{k_b}{k_g}{k_cost};
    mean_d_true = mean(d_true,3);
    std_d_true = std(d_true,0,3);
    
    % number of times of STAY across episodes (Fig 7B,C, 8A,B,D,E right)
    F = figure;
    A = axes;
    hold on
    if k_para == 6
        axis([0 21 0 12]);
    else
        axis([0 21 0 4]);
    end
    for k_state=1:num_state-1
        P = plot([1:20],mean_num_STAY(:,k_state)); set(P,'Color',(k_state*0.16)*[1 1 1],'LineWidth',2);
    end
    xlabel('Episode','FontSize',24);
    ylabel({'Number of times of','STAY (Procrastination)'},'FontSize',24);
    set(A,'XTick',[5:5:20],'XTickLabel',[5:5:20]);
    if k_para == 6
        set(A,'YTick',[0:2:12],'YTickLabel',[0:2:12]);
    else
        set(A,'YTick',[0:1:4],'YTickLabel',[0:1:4]);
    end
    set(A,'FontSize',20);
    if save_fig
        print(F,'-depsc',['Fig' tmp_fignamge_set{k_para} '_right']);
    end
    
    % approximated/true value difference at the 1st and the 20th episodes (Fig 7B,C, 8A,B,D,E left&middle)
    tmp_epi = [1 20];
    tmp_title = {'1st episode','20th episode'};
    for k = 1:2
        F = figure;
        A = axes;
        hold on;
        if k_para == 6
            axis([0 5 -0.1 0.6]);
        else
            axis([0 5 -0.1 0.4]);
        end
        P = plot([0 5],[0 0],'k:');
        P = errorbar(mean_d_est(tmp_epi(k),:),std_d_est(tmp_epi(k),:),'r','LineWidth',1);
        P = plot(mean_d_est(tmp_epi(k),:),'r','LineWidth',2);
        P = errorbar(mean_d_true(tmp_epi(k),:),std_d_true(tmp_epi(k),:),'k','LineWidth',1);
        P = plot(mean_d_true(tmp_epi(k),:),'k','LineWidth',2);
        title(tmp_title{k},'FontSize',24);
        xlabel('State','FontSize',24);
        ylabel({'Approximated / True','Value Difference: GO - STAY'},'FontSize',24);
        set(A,'Xtick',[1:4], 'Xticklabel',[1:4]);
        if k_para == 6
            set(A,'Ytick',[-0.1:0.1:0.6], 'Yticklabel',[-0.1:0.1:0.6]);
        else
            set(A,'Ytick',[-0.1:0.1:0.4], 'Yticklabel',[-0.1:0.1:0.4]);
        end
        set(A,'FontSize',20);
        if save_fig
            print(F,'-depsc',['Fig' tmp_fignamge_set{k_para} '_epi' num2str(tmp_epi(k))]);
        end
    end
    
end
