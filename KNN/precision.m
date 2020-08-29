features = {'dia'};
dist_metrics = {'l2'};

sets = {'train', 'test'};
datasets = ['espgame'];
test_image_count = [2081];
train_image_count = [18689];
dict_size = [268];

ids = 1;   

labels_per_image = 5;           %labels to be allotted per test image           
nearest_neighbours = 5;         %number of nearest neighbours considered per test image

%[test_annot] = get_test_annot();
%[train_annot] = get_train_annot();
%a = vec_read('espgame_test_annot.hvecs')
%b = vec_read('espgame_train_annot.hvecs')
espgame_test_annot=double(vec_read('espgame_test_annot.hvecs'));
espgame_train_annot=double(vec_read('espgame_train_annot.hvecs'));

espgame_label_train_freq = sum(espgame_test_annot);     

distf = load('espgame_dist.mat');
espgame_distances = distf.distances;

cooccur = (espgame_train_annot.')*espgame_train_annot;
test_labels = zeros(test_image_count(ids), dict_size(ids));
for i = 1:test_image_count(ids)
    distances = espgame_distances(i, :);
    
    [~, neighbours] = sort(distances);

    % Perform label transfer here
    labels = zeros(1, dict_size(ids));                % labels to be assigned to test image
    
    % Sorting labels for nearest neighbour wrt their frequencies in
    % training dataset
    nearest_nbr_labels = find(espgame_train_annot(neighbours(1), :));
    [~, label_freq_sort] = sort(espgame_label_train_freq(nearest_nbr_labels), 'descend');
    nearest_nbr_labels = nearest_nbr_labels(label_freq_sort);
        
    sz = numel(nearest_nbr_labels);
    if sz >= labels_per_image
        % assign first n labels to test image if nearest nbr labels are
        % more than n (n = labels_per_image)
        labels(nearest_nbr_labels(1:labels_per_image)) = 1;    
    else
        % if nearest nbr has less than n labels, assign all of them to
        % test image
        labels(nearest_nbr_labels(1:sz)) = 1;
        other_nbrs_annot = espgame_train_annot(neighbours(2:nearest_neighbours), :);
        local_labels_freq = sum(other_nbrs_annot);
        other_nbrs_labels = find(local_labels_freq);
        local_labels_cooccurrence = zeros(1, dict_size(ids));
        for lbl = 1:numel(other_nbrs_labels)
            if ismember(other_nbrs_labels(lbl), nearest_nbr_labels)
                continue; 
            end
            local_labels_cooccurrence(other_nbrs_labels(lbl)) = sum(cooccur(other_nbrs_labels(lbl), nearest_nbr_labels));
        end
        local_labels_priority = local_labels_freq .* local_labels_cooccurrence;
        transferrable_labels_cnt = numel(other_nbrs_labels);
        [~, other_lbls_sort] = sort(local_labels_priority, 'descend');
        labels(other_lbls_sort(1:min(labels_per_image-sz, transferrable_labels_cnt))) = 1;

        
    end
    test_labels(i, :) = labels;


end

mean_precision = 0;
mean_recall = 0;
n_plus = 0;
for l = 1:dict_size(ids)
    ground_truth = sum(espgame_test_annot(:, l));
    predicted = sum(test_labels(:, l));
    correct = sum(espgame_test_annot(1:test_image_count(ids), l) & test_labels(:, l));
    if correct > 0
        n_plus = n_plus + 1;
    end
    mean_precision = mean_precision + correct/(predicted+1e-10);
    mean_recall = mean_recall + correct/ground_truth;
end
mean_precision = 100*mean_precision/dict_size(ids);
mean_recall = 100*mean_recall/dict_size(ids);
f1_score = 2 * mean_precision * mean_recall / (mean_precision + mean_recall + 1e-10);