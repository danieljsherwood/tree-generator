function [number_leaves,number_branches]=tree(axes,h_leafbase,d_trunk,h_trunk,n_trunk_seg,ang_firstbranch,l_firstbranch,fac_crown_vertical,fac_verticaltaper,n_branch_generations,ang_branch_split,ang_branch_rot,l_branch_segment,n_branch_children,fac_branch_taper,const_gravity,speed_wind,bool_leaf,colour_leaf,size_leaf,n_leaf_vertices)

%% Setting up random numbers
% Set up a 10000 by 1 matrix where each element is a binomial between 0 and
% 1
randlist=rand(10000,1);

%% Setting up matrices with branch co-ordinates (rank 0-6)
'Plotting branches'
% Set up matrix for the trunk co-ordinates
p_0=[];
% Add the first section of the trunk below the leafbase
p_0(1,:)=[0,0,0,0,0,h_leafbase,d_trunk];
% Split the rest of the trunk height into even sections  based on n_trunk_seg
% and add one row to p_0 per section
for i=2:n_trunk_seg
    p_0(i,:)=[0,0,((i-2)*((h_trunk-h_leafbase)/(n_trunk_seg-1))+h_leafbase),0,0,((i-1)*((h_trunk-h_leafbase)/(n_trunk_seg-1))+h_leafbase),d_trunk];
end
% Draw the trunk on the specified axes
draw_branches(p_0,randlist,axes);
% Set up empty matrix for first branch co-ordinates
p_1=[];
% For each row in p_0, create rows in p_1 corresponding to the children of
% that branch
for i=1:size(p_0)
    % Define a point representing the end of the new branch
    [x,y,z]=newpoint(p_0(i,1),p_0(i,2),p_0(i,3),p_0(i,4),p_0(i,5),p_0(i,6),ang_firstbranch,i*140,l_firstbranch);
    % Append a row onto the p_1 matrix with the start and end points of the
    % new branch, as well as the thickness of the branch
    p_1(end+1,:)=[p_0(i,4),p_0(i,5),p_0(i,6),x,y,z,d_trunk/1.5];
end
% Decrease the x and y values of the top branch to make it shorter and
% more vertical
p_1(end,4:5)=p_1(end,4:5)/fac_crown_vertical;
% Draw the branches on the specified axes
draw_branches(p_1,randlist,axes);
% Define a new structure with a 'matrix' property, in which each successive
% matrix of points will be stored. Add p_0 and p_1 to this structure
points=struct('matrix',{p_0,p_1});
% For each successive generation of branch, from 2 up to
% n_branch_generations, add all the branch segments to a new matrix,
% defining the co-ordinates of each child based on the previous generation
for i=2:n_branch_generations
    % Define a new matrix within the "points" structure based on the
    % previous generation
    points(end+1).matrix=children(points(end).matrix,ang_branch_split,n_branch_children,l_branch_segment,ang_branch_rot,d_trunk,fac_branch_taper,h_leafbase,speed_wind,const_gravity,randlist,i);
    % Draw the branches of this new generation on the specified axes
    draw_branches(points(end).matrix,randlist,axes);
end

%% Plotting leaves
% Don't plot leaves if told not to
if bool_leaf~="No"
    'Plotting leaves'
    %For every branch in the final matrix of co-ordinates
    for i=1:size(points(end).matrix)
        % Set up a new matrix for leaf co-ordinates
        leaf=[];
        % For every vertex in this leaf
        for j=1:n_leaf_vertices
            % Define a point near the end of the branch, at the same
            % distance parallel to the branch from its origin, at a
            % distance away from the branch defined by size_leaf, and at a
            % rotation around the branch axis proportional to the number of
            % this vertex
            [x,y,z]=newpoint(points(end).matrix(i,1),points(end).matrix(i,2),points(end).matrix(i,3),points(end).matrix(i,4),points(end).matrix(i,5),points(end).matrix(i,6),90,j*360/n_leaf_vertices,size_leaf);
            % If leaves are specified to be fallen, set the z co-ordinate
            % of this point to be zero
            if bool_leaf=="Fallen"
                z=0;
            end
            % Append this point to the "leaf" matrix
            leaf(j,:)=[x,y,z];
        end
        % Set the colour of leaf according to parameter which sets the
        % range of colours available and a random number which specifies
        % the exact colour within this range
        if colour_leaf=="Light green"
            leafcolour=(hsl2rgb([130/100, (70+30*rand)/100, (50+20*rand)/100]));
        elseif colour_leaf=="Deep green"
            leafcolour=(hsl2rgb([130/100, (70+30*rand)/100, (20+30*rand)/100]));
        elseif colour_leaf=="Brown"
            leafcolour=(hsl2rgb([(0+20*rand)/100, (70+30*rand)/100, (30+30*rand)/100]));
        elseif colour_leaf=="Purple"
            leafcolour=(hsl2rgb([90/100, (50+30*rand)/100, (10+30*rand)/100]));
        elseif colour_leaf=="Muted green"
            leafcolour=(hsl2rgb([145/100, (55+10*rand)/100, (15+10*rand)/100]));
        end
        % Draw the leaf on the specified axes
        patch(axes,leaf(:,1),leaf(:,2),leaf(:,3),leafcolour);
    end
end

%% Stats
number_leaves=size(points(end).matrix,1);
number_branches=0;
for i=2:n_branch_generations+1
    number_branches=number_branches+size(points(i).matrix,1);
end
%% Custom functions

% Given a matrix containing the co-ordinates of a set of branches, produce
% a matrix with the co-ordinates of the children of each of these branches
    function [p_n]=children(p_m,ang_branch,n_branch_children,l_branch_seg,ang_rot,d_trunk,taper_rank,h_leafbase,eff_wind,eff_grav,randlist,rank)
    % Set up an empty matrix for the next generation of branches
    p_n=[];
    % For each branch in p_m
    for i=1:size(p_m)
        % Create new children up to the limit set by n_branch_children
        for j=1:n_branch_children
            % Define the endpoint of the child
            [x,y,z]=newpoint(p_m(i,1),p_m(i,2),p_m(i,3),p_m(i,4),p_m(i,5),p_m(i,6),ang_branch,randlist(i)*ang_rot+j*360/n_branch_children,l_branch_seg);
            % If z is lower than the value specified for leafbase height,
            % raise z to above the leafbase
            if z<h_leafbase
                z=h_leafbase+0.5;
            end
            % Move in the x direction according to the wind coefficient
            x=x+eff_wind/80;
            % Move in the z direction according to the gravitational effect
            z=z+0.1*(9.81-eff_grav);
            % Decrease the x and y co-ordinates in proportion to height,
            % depending on the strength of vertical taper specified
            x=x*(1-(z*fac_verticaltaper/100));
            y=y*(1-(z*fac_verticaltaper/100));
            % Append a row to the p_n matrix with the co-ordinates and
            % thickness of the child. Element 1-3 describes the x,y,z
            % co-ordinates of the child's startpoint, element 4-6 describe
            % the x,y,z co-ordinates of the child's endpoint, and element 7
            % specifies the branch thickness.
            p_n(end+1,:)=[p_m(i,4),p_m(i,5),p_m(i,6),x,y,z,d_trunk/rank^taper_rank];
        end
    end
    end

%Given two points, find the vector that connects them, and rotate
%it by angles alpha and beta with regards to itself, and define a new
%vector with specified length
    function [new_x,new_y,new_z]=newpoint(start_x,start_y,start_z,end_x,end_y,end_z,alpha,beta,length)
    % Define a vector based on the direction of the old branch
    vector_before=[end_x-start_x,end_y-start_y,end_z-start_z];
    % Unit vector of old branch
    vector_before_unit=vector_before/norm(vector_before);
    % Rotate this vector away from its axis by alpha
    vector_int=rotVecAroundArbAxis(vector_before_unit,vector_before_unit*rotx(90),alpha);
    % Unit vector of intermediate vector
    vector_int_unit=vector_int/norm(vector_int);
    % Rotate this vector around the original axis by beta
    vector_after=rotVecAroundArbAxis(vector_int_unit,vector_before_unit,beta);
    % Define a new position based on this vector and the original endpoint
    position_new=[end_x,end_y,end_z]+[length*vector_after];
    new_x=position_new(1);
    new_y=position_new(2);
    new_z=position_new(3);
    end
    
%Given a matrix of specified branches, plot the branches in varying
%colours of brown and with line thickness set accordingly
    function draw_branches(coordinates,randlist,axes)
    % For each branch, draw it in the specified colour
    for n=1:size(coordinates)
        rand=randlist(n);
        % Define a colour within the brown colourspace based on the random
        % number taken from randlist
        colour=hsl2rgb([5/100, (40+10*rand)/100, (10+20*rand)/100]);
        % Draw the branch on the specified axes
        plot3(axes,[coordinates(n,1),coordinates(n,4)],[coordinates(n,2),coordinates(n,5)],[coordinates(n,3),coordinates(n,6)],'Color',colour,'LineWidth',coordinates(n,7)*30);
    end
    end

%% Imported functions

    function rotatedUnitVector = rotVecAroundArbAxis(unitVec2Rotate,rotationAxisUnitVec,theta)
        %% Purpose:
        %
        %  This routine will allow a unit vector, unitVec2Rotate, to be rotated
        %  around an axis defined by the RotationAxisUnitVec.  This is performed by
        %  first rotating the unit vector around it's own cartesian axis (in this
        %  case we will rotate the vector around the z-axis, [0 0 1]) corresponding
        %  to each rotation angle specified by the user via the variable theta ...
        %  this rotated vector is then transformed around the user defined axis of
        %  rotation as defined by the rotationAxisUnitVec variable.
        %
        %
        %% References:
        %  Murray, G. Rotation About an Arbitrary Axis in 3 Dimensions. 06/06/2013.
        %  http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
        %
        %% Inputs:
        %  unitVec2Rotate               [N x 3]                 Unit Vector in
        %                                                       Cartesian
        %                                                       Coordinates to
        %                                                       rotate
        %                                                       [x,y,z]
        %
        %
        %  rotationAxisUnitVec          [N x 3]                 Unit Vector with
        %                                                       respect to the same
        %                                                       cartesian coordinates
        %                                                       used for unitVec2Rotate
        %                                                       [x,y,z]
        %
        %  theta                        [N x 1]                 Angle in degrees
        %                                                       in which to rotate
        %                                                       the unitVec2Rotate
        %                                                       about the Z-axis
        %                                                       before transforming
        %                                                       it to the
        %                                                       RotateionAxisUnitVec
        %                                                       This rotation is
        %                                                       counter clockwise
        %                                                       when theta is
        %                                                       positive, clockwise
        %                                                       when theta is
        %                                                       negative.
        %
        %% Outputs:
        %  rotatedUnitVector            [N x 3]                 Resulting vector
        %                                                       of rotating the
        %                                                       unitVec2Rotate
        %                                                       about the z-axis
        %                                                       described by the
        %                                                       angle theta, then
        %                                                       transforming the
        %                                                       rotated vectors
        %                                                       with respect to the
        %                                                       rotateionAxisUnitVec
        %
        %% Revision History:
        %  Darin C. Koblick                                        (c)  03-03-2015
        %
        %  Darin C. Koblick      Fixed order of rotations               07-30-2015
        %% ---------------------- Begin Code Sequence -----------------------------
        if nargin == 0
            unitVec2Rotate = [1 0 1]./norm([1 0 1]);
            rotationAxisUnitVec = [1 1 1]./norm([1 1 1]);
            theta = (0:5:360)';
            rotatedUnitVector = rotVecAroundArbAxis(unitVec2Rotate,rotationAxisUnitVec,theta);
            %Show a graphical representation of the rotated vector:
            figure('color',[1 1 1]);
            quiver3(zeros(numel(theta),1),zeros(numel(theta),1),zeros(numel(theta),1), ...
                rotatedUnitVector(:,1),rotatedUnitVector(:,2),rotatedUnitVector(:,3),'k','linewidth',2);
            quiver3(0,0,0,rotationAxisUnitVec(1), ...
                rotationAxisUnitVec(2), ...
                rotationAxisUnitVec(3), ...
                'r','linewidth',5);
            axis equal;
            return;
        end
        %Check the dimensions of the input vectors to see if we need to repmat
        %them:
        if size(unitVec2Rotate,1) == 1
            unitVec2Rotate = repmat(unitVec2Rotate,[numel(theta) 1]);
        end
        if size(rotationAxisUnitVec,1) == 1
            rotationAxisUnitVec = repmat(rotationAxisUnitVec,[numel(theta) 1]);
        end
        %% Step One: take the unit vector rotation axis and rotate into z:
        R2Z = vecRotMat(rotationAxisUnitVec,repmat([0 0 1],[size(rotationAxisUnitVec,1) 1]));
        unitVectortoRotateAboutZ =Dim33Multiply(unitVec2Rotate,R2Z);
        % Rotate the unit vector about the z-axis:
        rotatedAboutZAxisUnitVec = bsxRz(unitVectortoRotateAboutZ,theta.*pi/180);
        %% Step Two: Find the rotation Matrix to transform the z-axis to rotationAxisUnitVec
        R = vecRotMat(repmat([0 0 1],[size(rotationAxisUnitVec,1) 1]),rotationAxisUnitVec);
        %% Step Three: Apply the Rotation matrix to the rotatedAboutZAxisUnitVec vectors
        rotatedUnitVector =Dim33Multiply(rotatedAboutZAxisUnitVec,R);
    end
    function a = Dim33Multiply(a,b)
        % Purpose:
        % Given a, an [N x 3] matrix, use b, an [3 x 3 x N] rotation matrix to come
        % up with a vectorized solution to b*a
        %
        %% Inputs:
        %  a            [N x 3]                                        N x 3 vector
        %
        %  b            [3 x 3 x N]                                    3 x 3 x N
        %                                                              matrix
        %
        %% Outputs:
        %  a            [N x 3]                                        vectorized
        %                                                              solution
        %                                                              a = b*a
        %
        %% Revision History:
        % Created by Darin C. Koblick   (C)                                    2013
        %% ---------------------- Begin Code Sequence -----------------------------
        a =cat(1,sum(permute(bsxfun(@times,b(1,:,:),permute(a,[3 2 1])),[2 3 1]),1), ...
            sum(permute(bsxfun(@times,b(2,:,:),permute(a,[3 2 1])),[2 3 1]),1), ...
            sum(permute(bsxfun(@times,b(3,:,:),permute(a,[3 2 1])),[2 3 1]),1))';
    end
    function R = vecRotMat(f,t)
        %% Purpose:
        %Commonly, it is desired to have a rotation matrix which will rotate one
        %unit vector, f,  into another unit vector, t. It is desired to
        %find R(f,t) such that R(f,t)*f = t.
        %
        %This program, vecRotMat is the most
        %efficient way to accomplish this task. It uses no square roots or
        %trigonometric functions as they are very computationally expensive.
        %It is derived from the work performed by Moller and Hughes, which have
        %suggested that this method is the faster than any previous transformation
        %matrix methods tested.
        %
        %
        %% Inputs:
        %f                      [N x 3]                         N number of vectors
        %                                                       in which to
        %                                                       transform into
        %                                                       vector t.
        %
        %t                      [N x 3]                         N number of vectors
        %                                                       in which it is
        %                                                       desired to rotate
        %                                                       f.
        %
        %
        %% Outputs:
        %R                      [3 x 3 x N]                     N number of
        %                                                       rotation matrices
        %
        %% Source:
        % Moller,T. Hughes, F. "Efficiently Building a Matrix to Rotate One
        % Vector to Another", 1999. http://www.acm.org/jgt/papers/MollerHughes99
        %
        %% Created By:
        % Darin C. Koblick (C) 07/17/2012
        % Darin C. Koblick     04/22/2014       Updated when lines are close to
        %                                       parallel by checking
        %% ---------------------- Begin Code Sequence -----------------------------
        %It is assumed that both inputs are in vector format N x 3
        dim3 = 2;
        %Declare function handles for multi-dim operations
        normMD = @(x,y) sqrt(sum(x.^2,y));
        anyMD  = @(x) any(x(:));
        % Inputs Need to be in Unit Vector Format
        if anyMD(single(normMD(f,dim3)) ~= single(1)) || anyMD(single(normMD(t,dim3)) ~= single(1))
            error('Input Vectors Must Be Unit Vectors');
        end
        %Pre-Allocate the 3-D transformation matrix
        R = NaN(3,3,size(f,1));
        v = permute(cross(f,t,dim3),[3 2 1]);
        c = permute(dot(f,t,dim3),[3 2 1]);
        h = (1-c)./dot(v,v,dim3);
        idx  = abs(c) > 1-1e-13;
        %If f and t are not parallel, use the following computation
        if any(~idx)
            %For any vector u, the rotation matrix is found from:
            R(:,:,~idx) = ...
                [c(:,:,~idx) + h(:,:,~idx).*v(:,1,~idx).^2,h(:,:,~idx).*v(:,1,~idx).*v(:,2,~idx)-v(:,3,~idx),h(:,:,~idx).*v(:,1,~idx).*v(:,3,~idx)+v(:,2,~idx); ...
                h(:,:,~idx).*v(:,1,~idx).*v(:,2,~idx)+v(:,3,~idx),c(:,:,~idx)+h(:,:,~idx).*v(:,2,~idx).^2,h(:,:,~idx).*v(:,2,~idx).*v(:,3,~idx)-v(:,1,~idx); ...
                h(:,:,~idx).*v(:,1,~idx).*v(:,3,~idx)-v(:,2,~idx),h(:,:,~idx).*v(:,2,~idx).*v(:,3,~idx)+v(:,1,~idx),c(:,:,~idx)+h(:,:,~idx).*v(:,3,~idx).^2];
        end
        %If f and t are close to parallel, use the following computation
        if any(idx)
            f = permute(f,[3 2 1]);
            t = permute(t,[3 2 1]);
            p = zeros(size(f));
            iidx = abs(f(:,1,:)) <= abs(f(:,2,:)) & abs(f(:,1,:)) < abs(f(:,3,:));
            if any(iidx & idx)
                p(:,1,iidx & idx) = 1;
            end
            iidx = abs(f(:,2,:)) < abs(f(:,1,:)) & abs(f(:,2,:)) <= abs(f(:,3,:));
            if any(iidx & idx)
                p(:,2,iidx & idx) = 1;
            end
            iidx = abs(f(:,3,:)) <= abs(f(:,1,:)) & abs(f(:,3,:)) < abs(f(:,2,:));
            if any(iidx & idx)
                p(:,3,iidx & idx) = 1;
            end
            u = p(:,:,idx)-f(:,:,idx);
            v = p(:,:,idx)-t(:,:,idx);
            rt1 = -2./dot(u,u,dim3);
            rt2 = -2./dot(v,v,dim3);
            rt3 = 4.*dot(u,v,dim3)./(dot(u,u,dim3).*dot(v,v,dim3));
            R11 = 1 + rt1.*u(:,1,:).*u(:,1,:)+rt2.*v(:,1,:).*v(:,1,:)+rt3.*v(:,1,:).*u(:,1,:);
            R12 = rt1.*u(:,1,:).*u(:,2,:)+rt2.*v(:,1,:).*v(:,2,:)+rt3.*v(:,1,:).*u(:,2,:);
            R13 = rt1.*u(:,1,:).*u(:,3,:)+rt2.*v(:,1,:).*v(:,3,:)+rt3.*v(:,1,:).*u(:,3,:);
            R21 = rt1.*u(:,2,:).*u(:,1,:)+rt2.*v(:,2,:).*v(:,1,:)+rt3.*v(:,2,:).*u(:,1,:);
            R22 = 1 + rt1.*u(:,2,:).*u(:,2,:)+rt2.*v(:,2,:).*v(:,2,:)+rt3.*v(:,2,:).*u(:,2,:);
            R23 = rt1.*u(:,2,:).*u(:,3,:)+rt2.*v(:,2,:).*v(:,3,:)+rt3.*v(:,2,:).*u(:,3,:);
            R31 = rt1.*u(:,3,:).*u(:,1,:)+rt2.*v(:,3,:).*v(:,1,:)+rt3.*v(:,3,:).*u(:,1,:);
            R32 = rt1.*u(:,3,:).*u(:,2,:)+rt2.*v(:,3,:).*v(:,2,:)+rt3.*v(:,3,:).*u(:,2,:);
            R33 = 1 + rt1.*u(:,3,:).*u(:,3,:)+rt2.*v(:,3,:).*v(:,3,:)+rt3.*v(:,3,:).*u(:,3,:);
            R(:,:,idx) = [R11 R12 R13; R21 R22 R23; R31 R32 R33];
        end
    end
    function m = bsxRz(m,theta)
        %% Purpose:
        % Perform a rotation of theta radians about the z-axis on the vector(s)
        % described by m.
        %
        %% Inputs:
        % m         [N x 3]                                         vector matrix
        %                                                           in which you
        %                                                           would like to
        %                                                           rotate with the
        %                                                           x,y,z
        %                                                           components
        %                                                           specified along
        %                                                           a specific
        %                                                           dimension
        %
        % theta     [N x 1]                                         Rotation Angle
        %                                                           about z-axis
        %                                                           in radians
        %
        %% Outputs:
        % m         [N x 3]
        %
        %% Revision History:
        %  Darin C Koblick (C)                              Initially Created 2013
        %% ---------------------- Begin Code Sequence -----------
        %Assemble the rotation matrix
        Rz = zeros(3,3,size(m,1));
        Rz(1,1,:) = cos(theta);  Rz(1,2,:) = -sin(theta);
        Rz(2,1,:) = sin(theta);  Rz(2,2,:) =  cos(theta);
        Rz(3,3,:) = 1;
        %Dim33Multiply
        m = Dim33Multiply(m,Rz);
    end
    function rgb=hsl2rgb(hsl_in)
        %Converts Hue-Saturation-Luminance Color value to Red-Green-Blue Color value
        %
        %Usage
        %       RGB = hsl2rgb(HSL)
        %
        %   converts HSL, a M [x N] x 3 color matrix with values between 0 and 1
        %   into RGB, a M [x N] X 3 color matrix with values between 0 and 1
        %
        %See also rgb2hsl, rgb2hsv, hsv2rgb
        % (C) Vladimir Bychkovsky, June 2008
        % written using:
        % - an implementation by Suresh E Joel, April 26,2003
        % - Wikipedia: http://en.wikipedia.org/wiki/HSL_and_HSV
        hsl=reshape(hsl_in, [], 3);
        H=hsl(:,1);
        S=hsl(:,2);
        L=hsl(:,3);
        lowLidx=L < (1/2);
        q=(L .* (1+S) ).*lowLidx + (L+S-(L.*S)).*(~lowLidx);
        p=2*L - q;
        hk=H; % this is already divided by 360
        t=zeros([length(H), 3]); % 1=R, 2=B, 3=G
        t(:,1)=hk+1/3;
        t(:,2)=hk;
        t(:,3)=hk-1/3;
        underidx=t < 0;
        overidx=t > 1;
        t=t+underidx - overidx;
        
        range1=t < (1/6);
        range2=(t >= (1/6) & t < (1/2));
        range3=(t >= (1/2) & t < (2/3));
        range4= t >= (2/3);
        % replicate matricies (one per color) to make the final expression simpler
        P=repmat(p, [1,3]);
        Q=repmat(q, [1,3]);
        rgb_c= (P + ((Q-P).*6.*t)).*range1 + ...
            Q.*range2 + ...
            (P + ((Q-P).*6.*(2/3 - t))).*range3 + ...
            P.*range4;
        
        rgb_c=round(rgb_c.*10000)./10000;
        rgb=reshape(rgb_c, size(hsl_in));
    end


end