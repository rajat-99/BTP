function  [results] = parameters_cal(test_labels)


% sets = {'train', 'test'};
% datasetsCap = {'Corel5k', 'ESPGame', 'IAPRTC12'};
% datasets = {'corel5k', 'espgame', 'iaprtc12'};
test_image_count = [499 2081 1962];
train_image_count = [4500 18689 17665];
dict_size = [260 268 291];

ids = 2;

load('espgame_semantic_hierarchy_structure.mat');
espgame_test = full(semantic_hierarchy_structure.label_test_SH_augmented);


mean_precision = 0;
mean_recall = 0;
n_plus = 0;
%for l = 1:dict_size(ids)
%    ground_truth = sum(espgame_test(:, l));
%    predicted = sum(test_labels(:, l));
%    correct = sum(espgame_test(1:test_image_count(ids), l) & test_labels(:, l));
%    if correct > 0
%        n_plus = n_plus + 1;
%    end
%    mean_precision = mean_precision + correct/(predicted+1e-10);
%    mean_recall = mean_recall + correct/ground_truth;
%end

%mean_precision = 100*mean_precision/dict_size(ids);
%mean_recall = 100*mean_recall/dict_size(ids);
%f1_score = 2 * mean_precision * mean_recall / (mean_precision + mean_recall + 1e-10);
%results = [mean_precision mean_recall f1_score n_plus];
%f1_score = 0;
[semantic_precision,semantic_recall,semantic_f1] = semantic(dict_size(ids),test_labels,espgame_test);
%results = [mean_precision,mean_recall,f1_score,n_plus,semantic_precision,semantic_recall,semantic_f1];
results = [semantic_precision,semantic_recall,semantic_f1];


%T = table(mean_precision,mean_recall,f1_score,n_plus,semantic_precision,semantic_recall,semantic_f1);
%writetable(T,'results1.txt');

% save([datasets{ids} '_results_p1.mat'], 'results');
end % of funtion