clear all
% set the seed of random number
rand_seed = 210701;
rand('twister',rand_seed);

% parameters
g = 0.85; % time discount factor
x = [g^4 g^3 g^2 g 1]; % feature variables for [S1 S2 S3 S4 S5]
n = 20; % number of episodes (trials)
num_state = 5; % number of states
a_SR = 0.05; % learning rate for reduced SR
m = 10000;
x_set = NaN(n,num_state,m);

% cost and reward
cost = 0.1;
costs = cost*ones(1,num_state-1);
rewards = [-costs 1];

% estimated true value of GO and STAY
v_GO = NaN(n,num_state-1,m);
v_STAY = NaN(n,num_state-1,m);

% approximated value of GO and STAY
v_GO2 = NaN(n,num_state-1,m);
v_STAY2 = NaN(n,num_state-1,m);

% estimated true state value
state_value = NaN(n,num_state,m);

% weight
w_history = NaN(n,num_state);
w_history2 = NaN(n,num_state,m);

% number of STAY
num_STAY = zeros(n,num_state-1,m);

% School term (episodes without procrastination)
w = 0;
for k_episode = 1:n
    a = (0.5/(1+0.2*k_episode)); % learning rate
    for k_state = 1:num_state-1
        TDerror = rewards(k_state) + g*w*x(k_state+1) - w*x(k_state);
        w = w + a*TDerror*x(k_state);
        w_history(k_episode,k_state) = w;
    end
    % at the last (goal) state
    k_state = num_state;
    TDerror = rewards(k_state) + 0 - w*x(k_state); % because there is no "next state"
    w = w + a*TDerror*x(k_state);
    w_history(k_episode,k_state) = w;
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
    fprintf('%d\n',k);
    x = [g^4 g^3 g^2 g 1]; % reset the feature variable
    w_2 = w;
    
    state_value_current = state_value_initial;
    
    for k_episode = 1:n
        a = (0.5/(1+0.2*k_episode)); % learning rate
        b = 20;
        
        for k_state = 1:num_state-1
            Vest_STAY = g*w_2*x(k_state);
            Vest_GO = g*w_2*x(k_state+1) - costs(k_state);
            Prob_STAY = choice_prob(Vest_STAY,Vest_GO,b);
            
            v_GO2(k_episode,k_state,k) = Vest_GO;
            v_STAY2(k_episode,k_state,k) = Vest_STAY;
            v_GO(k_episode,k_state,k) = g*state_value_current(k_state+1) - costs(k_state);
            v_STAY(k_episode,k_state,k) = g*state_value_current(k_state);
            
            %codes for STAY
            while rand < Prob_STAY % choose STAY with probability of Prob_STAY
                TDerror = 0 + g*w_2*x(k_state) - w_2*x(k_state);
                w_2 = w_2 + a*TDerror*x(k_state);
                num_STAY(k_episode,k_state,k) = num_STAY(k_episode,k_state,k)+1;
                
                TDerror_true = 0 + g*state_value_current(k_state) - state_value_current(k_state);
                state_value_current(k_state) = state_value_current(k_state) + a*TDerror_true;
                
                TDE_SR = 0 + g*x(k_state) - x(k_state);
                x(k_state) = x(k_state) + a_SR*TDE_SR;
                
                Vest_STAY = g*w_2*x(k_state);
                Vest_GO = g*w_2*x(k_state+1) - costs(k_state);
                Prob_STAY = choice_prob(Vest_STAY,Vest_GO,b);
            end
            
            %codes for GO
            TDerror = rewards(k_state) + g*w_2*x(k_state+1) - w_2*x(k_state);
            w_2 = w_2 + a*TDerror*x(k_state);
            
            TDerror_true = rewards(k_state) + g*state_value_current(k_state+1) - state_value_current(k_state);
            state_value_current(k_state) = state_value_current(k_state) + a*TDerror_true;
            w_history2(k_episode,k_state,k) = w_2;
            
            TDE_SR = 0 + g*x(k_state+1) - x(k_state);
            x(k_state) = x(k_state) + a_SR*TDE_SR;
        end
        
        % at the last (goal) state
        k_state = num_state;
        TDerror = rewards(k_state) + 0 - w_2*x(k_state); % because there is no "next state"
        w_2 = w_2 + a*TDerror*x(k_state);
        w_history2(k_episode,k_state,k) = w_2;
        
        state_value(k_episode,:,k) = state_value_current;
        x_set(k_episode,:,k) = x;
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
data.state_value_initial = state_value_initial;
data.x_set = x_set;
save procra_data_nonrigid data

%
load procra_data_nonrigid
g = 0.85; % time discount factor
n = 20; % number of episodes (trials)
num_state = 5; % number of states
m = 10000;

% case with a fixed cost
mean_num_STAY = mean(data.num_STAY,3);
std_num_STAY = std(data.num_STAY,0,3);
d_est = data.v_GO2 - data.v_STAY2;
mean_d_est = mean(d_est,3);
std_d_est = std(d_est,0,3);
d_true = data.v_GO - data.v_STAY;
mean_d_true = mean(d_true,3);
std_d_true = std(d_true,0,3);
mean_x_set = mean(data.x_set,3);

% make figures
save_fig = 1; % 1:save figures, 0:not save figures

% number of times of STAY across episodes (Fig 9C right)
F = figure;
A = axes;
hold on
axis([0 21 0 4]);
for k_state=1:num_state-1
    P = plot([1:20],mean_num_STAY(:,k_state)); set(P,'Color',(k_state*0.16)*[1 1 1],'LineWidth',2);
end
xlabel('Episode','FontSize',24);
ylabel({'Number of times of','STAY (Procrastination)'},'FontSize',24);
set(A,'XTick',[5:5:20],'XTickLabel',[5:5:20]);
set(A,'YTick',[0:1:4],'YTickLabel',[0:1:4]);
set(A,'FontSize',20);
if save_fig
    print(F,'-depsc','Fig9Cright');
end

% approximated/true value difference at the 20th episodes (Fig 9C middle)
F = figure;
A = axes;
hold on;
axis([0 5 -0.1 0.4]);
P = plot([0 5],[0 0],'k:');
P = errorbar(mean_d_est(20,:),std_d_est(20,:),'r','LineWidth',1);
P = plot(mean_d_est(20,:),'r','LineWidth',2);
P = errorbar(mean_d_true(20,:),std_d_true(20,:),'k','LineWidth',1);
P = plot(mean_d_true(20,:),'k','LineWidth',2);
title('20th episode','FontSize',24);
xlabel('State','FontSize',24);
ylabel({'Approximated / True','Value Difference: GO - STAY'},'FontSize',24);
set(A,'Xtick',[1:4], 'Xticklabel',[1:4]);
set(A,'Ytick',[-0.1:0.1:0.4], 'Yticklabel',[-0.1:0.1:0.4]);
set(A,'FontSize',20);
if save_fig
    print(F,'-depsc','Fig9Cmiddle');
end

% approximated/true value difference at the 1st episodes (Fig 9C left)
F = figure;
A = axes;
hold on;
axis([0 5 -0.1 0.4]);
P = plot([0 5],[0 0],'k:');
P = errorbar(mean_d_est(1,:),std_d_est(1,:),'r','LineWidth',1);
P = plot(mean_d_est(1,:),'r','LineWidth',2);
P = errorbar(mean_d_true(1,:),std_d_true(1,:),'k','LineWidth',1);
P = plot(mean_d_true(1,:),'k','LineWidth',2);
title('1st episode','FontSize',24);
xlabel('State','FontSize',24);
ylabel({'Approximated / True','Value Difference: GO - STAY'},'FontSize',24);
set(A,'Xtick',[1:4], 'Xticklabel',[1:4]);
set(A,'Ytick',[-0.1:0.1:0.4], 'Yticklabel',[-0.1:0.1:0.4]);
set(A,'FontSize',20);
if save_fig
    print(F,'-depsc','Fig9Cleft');
end
