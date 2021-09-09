clear all
% set the seed of random number
rand_seed = 210115;
rand('twister',rand_seed);

% parameters
g = 0.85; % time discount factor
x = [g^4 g^3 g^2 g 1]; % feature variables for [S1 S2 S3 S4 S5]
n = 20; % number of episodes (trials)
num_state = 5; % number of states
m = 10000; % number of simulations

% cost and reward
cost_set = [0:0.01:0.15];
for k_cost = 1:length(cost_set)
    cost = cost_set(k_cost);
    costs = cost*ones(1,num_state-1);
    rewards = [-costs 1];
    
    % estimated true value of GO and STAY
    v_GO{k_cost} = NaN(n,num_state-1,m);
    v_STAY{k_cost} = NaN(n,num_state-1,m);
    
    % approximated value of GO and STAY
    v_GO2{k_cost} = NaN(n,num_state-1,m);
    v_STAY2{k_cost} = NaN(n,num_state-1,m);
    
    % estimated true state value
    state_value{k_cost} = NaN(n,num_state,m);
    
    % weight
    w_history{k_cost} = NaN(n,num_state);
    w_history2{k_cost} = NaN(n,num_state,m);
    
    % number of STAY
    num_STAY{k_cost} = zeros(n,num_state-1,m);
    
    % School term (episodes without procrastination)
    w = 0;
    for k_episode = 1:n
        a = (0.5/(1+0.2*k_episode)); % learning rate
        for k_state = 1:num_state-1
            TDerror = rewards(k_state) + g*w*x(k_state+1) - w*x(k_state);
            w = w + a*TDerror*x(k_state);
            w_history{k_cost}(k_episode,k_state) = w;
        end
        % at the last (goal) state
        k_state = num_state;
        TDerror = rewards(k_state) + 0 - w*x(k_state); % because there is no "next state"
        w = w + a*TDerror*x(k_state);
        w_history{k_cost}(k_episode,k_state) = w;
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
        fprintf('%d-%d\n',k_cost,k);
        w_2 = w;
        
        state_value_current = state_value_initial;
        
        for k_episode = 1:n
            a = (0.5/(1+0.2*k_episode)); % learning rate
            b = 20;
            
            for k_state = 1:num_state-1
                Vest_STAY = g*w_2*x(k_state);
                Vest_GO = g*w_2*x(k_state+1) - costs(k_state);
                Prob_STAY = choice_prob(Vest_STAY,Vest_GO,b);
                
                v_GO2{k_cost}(k_episode,k_state,k) = Vest_GO;
                v_STAY2{k_cost}(k_episode,k_state,k) = Vest_STAY;
                v_GO{k_cost}(k_episode,k_state,k) = g*state_value_current(k_state+1) - costs(k_state);
                v_STAY{k_cost}(k_episode,k_state,k) = g*state_value_current(k_state);
                
                %codes for STAY
                while rand < Prob_STAY % choose STAY with probability of Prob_STAY
                    TDerror = 0 + g*w_2*x(k_state) - w_2*x(k_state);
                    w_2 = w_2 + a*TDerror*x(k_state);
                    num_STAY{k_cost}(k_episode,k_state,k) = num_STAY{k_cost}(k_episode,k_state,k)+1;
                    
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
                w_history2{k_cost}(k_episode,k_state,k) = w_2;
            end
            
            % at the last (goal) state
            k_state = num_state;
            TDerror = rewards(k_state) + 0 - w_2*x(k_state); % because there is no "next state"
            w_2 = w_2 + a*TDerror*x(k_state);
            w_history2{k_cost}(k_episode,k_state,k) = w_2;
            
            state_value{k_cost}(k_episode,:,k) = state_value_current;
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
save procra_data data

% make figures
save_fig = 1; % 1:save figures, 0:not save figures

%
load procra_data
g = 0.85; % time discount factor
x = [g^4 g^3 g^2 g 1]; % feature variables for [S1 S2 S3 S4 S5]
n = 20; % number of episodes (trials)
num_state = 5; % number of states
m = 10000;
cost_set = [0:0.01:0.15];

% case with a fixed cost
k_cost = 11;
cost = cost_set(k_cost);
costs = cost*ones(1,num_state-1);
rewards = [-costs 1];
mean_num_STAY = mean(data.num_STAY{k_cost},3);
std_num_STAY = std(data.num_STAY{k_cost},0,3);
d_est = data.v_GO2{k_cost} - data.v_STAY2{k_cost};
mean_d_est = mean(d_est,3);
std_d_est = std(d_est,0,3);
d_true = data.v_GO{k_cost} - data.v_STAY{k_cost};
mean_d_true = mean(d_true,3);
std_d_true = std(d_true,0,3);

% figures for w

% color matrix (Fig 2A, 4A)
image_scale = 80;
tmp_data{1} = data.w_history{k_cost};
tmp_data{2} = mean(data.w_history2{k_cost},3);
tmp_title = {'School Term','Vacation'};
tmp_fignames = {'Fig2A','Fig4A'};
for k = 1:2
    F = figure;
    A = axes;
    P = image(tmp_data{k}*image_scale);
    P = colorbar;
    title(tmp_title{k},'FontSize',24);
    xlabel('State','FontSize',24);
    ylabel('Episode','FontSize',24);
    set(A,'PlotBoxAspectRatio',[1 2 1],'XTick',[1:5],'YTick',[5:5:20],'FontSize',20);
    set(P,'YTick',[16:16:64],'YTickLabel',[16:16:64]/image_scale);
    if save_fig
        print(F,'-depsc',tmp_fignames{k});
    end
end

% line graph (Fig 2B, 4B)
tmp_data{1} = data.w_history{k_cost};
tmp_data{2} = mean(data.w_history2{k_cost},3);
tmp_title = {'School Term','Vacation'};
tmp_fignames = {'Fig2B','Fig4B'};
for k1 = 1:2
    F = figure;
    A = axes;
    hold on;
    axis([0 21 -0.2 0.8]);
    P = plot([0 21],[0 0],'k:');
    for k2 = 1:size(tmp_data{k1},2)
        P = plot(tmp_data{k1}(:,k2)); set(P,'Color',(k2*0.16)*[1 1 1],'LineWidth',2);
    end
    title(tmp_title{k1},'FontSize',24);
    xlabel('Episode','FontSize',24);
    ylabel({'Coefficient w of','Approximated Value Function'},'FontSize',24);
    set(A,'XTick',[5:5:20],'XTickLabel',[5:5:20]);
    set(A,'YTick',[-0.2:0.2:0.8],'YTickLabel',[-0.2:0.2:0.8]);
    set(A,'FontSize',20);
    if save_fig
        print(F,'-depsc',tmp_fignames{k1});
    end
end

% figures for number of times of STAY

% number of times of STAY across episodes (Fig 5C)
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
    print(F,'-depsc','Fig5C');
end

% number of times of STAY at the 1st and the 20th episodes (Fig 3B, 5D)
tmp_epi = [1 20];
tmp_title = {'1st episode','20th episode'};
tmp_fignames = {'Fig3B','Fig5D'};
for k = 1:2
    F = figure;
    A = axes;
    hold on
    axis([0 5 0 4]);
    P = plot(mean_num_STAY(tmp_epi(k),:),'kx-','LineWidth',2,'MarkerSize',20);
    title(tmp_title{k},'FontSize',24);
    xlabel('State','FontSize',24);
    ylabel({'Number of times of','STAY (Procrastination)'},'FontSize',24);
    set(A,'Xtick',[1 2 3 4],'XTickLabel',[1 2 3 4]);
    set(A,'YTick',[0:4],'YTickLabel',[0:4]);
    set(A,'FontSize',20);
    if save_fig
        print(F,'-depsc',tmp_fignames{k});
    end
end

% histograms for the number of times of STAY at the 1st and the 20th episodes (Fig 3C, 5E)
tmp_epi = [1 20];
tmp_title = {'1st episode','20th episode'};
tmp_fignames = {'Fig3C','Fig5E'};
xmax = ceil(max(max(max(data.num_STAY{k_cost}(tmp_epi,:,:))))/10)*10;
tmp_hist = NaN(num_state-1,xmax+1,length(tmp_epi));
for k = 1:length(tmp_epi)
    for k_state = 1:num_state-1
        tmp_hist(k_state,:,k) = hist(reshape(data.num_STAY{k_cost}(tmp_epi(k),k_state,:),1,m),[0:xmax]);
    end
end
ymax = ceil(max(max(max(tmp_hist)))/500)*500;
for k = 1:length(tmp_epi)
    F = figure;
    A = axes;
    hold on
    axis([-0.5 xmax+0.5 0 ymax]);
    for k_state = 1:num_state-1
        P = plot([0:xmax],tmp_hist(k_state,:,k)); set(P,'Color',(k_state*0.16)*[1 1 1],'LineWidth',2);
    end
    title(tmp_title{k},'FontSize',24);
    xlabel('Number of times of STAY','FontSize',24);
    ylabel('Frequency','FontSize',24);
    set(A,'Xtick',[0:5:xmax],'XTickLabel',[0:5:xmax]);
    set(A,'YTick',[0:1000:ymax],'YTickLabel',[0:1000:ymax]);
    set(A,'FontSize',20);
    if save_fig
        print(F,'-depsc',tmp_fignames{k});
    end
end

% figures for value differences  
    
% approximated/true value difference across episodes (Fig 5A)
for k_state=1:num_state-1
    F = figure;
    A = axes;
    hold on
    axis([0 21 -0.1 0.4]);
    P = plot([0 21],[0 0],'k:');
    P = errorbar([1:20],mean_d_true(:,k_state),std_d_true(:,k_state),'k','LineWidth',1);
    P = plot([1:20],mean_d_true(:,k_state),'k','LineWidth',2);
    P = errorbar([1:20],mean_d_est(:,k_state),std_d_est(:,k_state),'r','LineWidth',1);
    P = plot([1:20],mean_d_est(:,k_state),'r','LineWidth',2);
    title(['State ' num2str(k_state)],'FontSize',24);
    xlabel('Episode','FontSize',24);
    ylabel({'Approximated / True','Value Difference: GO - STAY'},'FontSize',24);
    set(A,'XTick',[5:5:20],'XTickLabel',[5:5:20]);
    set(A,'YTick',[-0.1:0.1:0.4],'YTickLabel',[-0.1:0.1:0.4]);
    set(A,'FontSize',20);
    if save_fig
        print(F,'-depsc',['Fig5A_S' num2str(k_state)]);
    end
end

% approximated/true value difference at the 20th episode (Fig 5B)
tmp_epi = 20;
tmp_title = '20th episode';
tmp_figname = 'Fig5B';
F = figure;
A = axes;
hold on;
axis([0 5 -0.1 0.4]);
P = plot([0 5],[0 0],'k:');
P = errorbar(mean_d_est(tmp_epi,:),std_d_est(tmp_epi,:),'r','LineWidth',1);
P = plot(mean_d_est(tmp_epi,:),'r','LineWidth',2);
P = errorbar(mean_d_true(tmp_epi,:),std_d_true(tmp_epi,:),'k','LineWidth',1);
P = plot(mean_d_true(tmp_epi,:),'k','LineWidth',2);
title(tmp_title,'FontSize',24);
xlabel('State','FontSize',24);
ylabel({'Approximated / True','Value Difference: GO - STAY'},'FontSize',24);
set(A,'Xtick',[1:4], 'Xticklabel',[1:4]);
set(A,'Ytick',[-0.1:0.1:0.4], 'Yticklabel',[-0.1:0.1:0.4]);
set(A,'FontSize',20);
if save_fig
    print(F,'-depsc',tmp_figname);
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
tmp_true_Go = g * state_value_initial(2:end) - cost;
tmp_true_STAY = g * state_value_initial(1:end-1);
tmp_true_Go_minus_STAY = tmp_true_Go - tmp_true_STAY;

% estimated state value at the beginning of vacation
est_state_value_beginning = data.w_history{k_cost}(end,end) * x;
tmp_est_Go = g * est_state_value_beginning(2:end) - cost;
tmp_est_STAY = g * est_state_value_beginning(1:end-1);
tmp_est_Go_minus_STAY = tmp_est_Go - tmp_est_STAY;

% approximated/true value difference at the beginning of the vacation (Fig 3Aa)
F = figure;
A = axes;
hold on;
axis([0 5 -0.1 0.2]);
P = plot([0 5],[0 0],'k:');
P = plot(tmp_est_Go_minus_STAY,'r','LineWidth',2);
P = plot(tmp_true_Go_minus_STAY,'k','LineWidth',2);
title('Beginning of the vacation','FontSize',24);
xlabel('State','FontSize',24);
ylabel({'Approximated / True','Value Difference: GO - STAY'},'FontSize',24);
set(A,'Xtick',[1:4], 'Xticklabel',[1:4]);
set(A,'Ytick',[-0.1:0.1:0.2], 'Yticklabel',[-0.1:0.1:0.2]);
set(A,'FontSize',20);
print(F,'-depsc','Fig3Aa');

% approximated/true value difference at the 1st and the 20th episodes (Fig 3Ab)
F = figure;
A = axes;
hold on;
axis([0 5 -0.1 0.2]);
P = plot([0 5],[0 0],'k:');
P = errorbar(mean_d_est(1,:),std_d_est(1,:),'r','LineWidth',1);
P = plot(mean_d_est(1,:),'r','LineWidth',2);
title('During the 1st episode','FontSize',24);
xlabel('State','FontSize',24);
ylabel({'Approximated','Value Difference: GO - STAY'},'FontSize',24);
set(A,'Xtick',[1:4], 'Xticklabel',[1:4]);
set(A,'Ytick',[-0.1:0.1:0.2], 'Yticklabel',[-0.1:0.1:0.2]);
set(A,'FontSize',20);
print(F,'-depsc','Fig3Ab');

% intuitive mechanism (Fig 7A)
w = data.w_history{k_cost}(end,end);
F = figure;
A = axes;
hold on
axis([0 num_state+1 0 1]);
P = plot([1:num_state],x,'r--','LineWidth',1);
P = plot([1:num_state],state_value_initial,'k','LineWidth',2);
P = plot([1:num_state],w*x,'r','LineWidth',2);
xlabel('State','FontSize',24);
ylabel({'Approximated / True',' State Value'},'FontSize',24);
set(A,'XTick',[1:1:5],'XTickLabel',[1:1:5]);
set(A,'YTick',[0:0.1:1],'YTickLabel',[0:0.1:1]);
set(A,'FontSize',20);
print(F,'-depsc','Fig7A');

% varying the cost (Fig 6)
tmp_title = {'1st episode','20th episode'};
tmp_fignames = {'Fig6A','Fig6B'};
mean_num_STAY_1st_20th{1} = NaN(length(cost_set),num_state-1);
mean_num_STAY_1st_20th{2} = NaN(length(cost_set),num_state-1);
for k_cost = 1:length(cost_set)
    tmp = mean(data.num_STAY{k_cost},3);
    mean_num_STAY_1st_20th{1}(k_cost,:) = tmp(1,:);
    mean_num_STAY_1st_20th{2}(k_cost,:) = tmp(20,:);
end
for k1 = 1:2
    F = figure;
    A = axes;
    hold on;
    axis([cost_set(1) cost_set(end) 0 18]);
    for k2 = 1:4
        P = plot(cost_set,mean_num_STAY_1st_20th{k1}(:,k2)); set(P,'Color',(k2*0.16)*[1 1 1],'LineWidth',2);
    end
    title(tmp_title{k1},'FontSize',24);
    xlabel('Cost','FontSize',24);
    ylabel({'Number of times of','STAY (Procrastination)'},'FontSize',24);
    set(A,'Xtick',[0:0.05:0.15], 'Xticklabel',[0:0.05:0.15]);
    set(A,'Ytick',[0:2:18], 'Yticklabel',[0:2:18]);
    set(A,'FontSize',20);
    if save_fig
        print(F,'-depsc',tmp_fignames{k1});
    end
end
